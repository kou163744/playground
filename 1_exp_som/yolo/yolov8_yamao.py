#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import os.path as osp
import random
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import roslib
import rospy
import torch
from geometry_msgs.msg import Point, Pose, Quaternion
from image_geometry import PinholeCameraModel
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Header
from tam_object_detection.msg import BBox, ObjectDetection, Pose3D
from tamlib.cv_bridge import CvBridge
from tamlib.node_template import Node
from tamlib.open3d import Open3D
from tamlib.tf import Transform
from torch import Tensor

sys.path.append(
    osp.join(
        roslib.packages.get_pkg_dir("tam_object_detection"),
        "../third_party/YOLOv8-TensorRT/",
    )
)
from config import MASK_COLORS as yolov8_msk_clr, ALPHA as yolov8_alp
from models import utils as yolov8_util
from models import torch_utils as yolov8_torch
from models import TRTModule


class YOLOv8TensorRT(Node):
    def __init__(self) -> None:
        super().__init__()

        # Parameters
        self.p_action_name = rospy.get_param("~action_name", "object_detection")
        p_class_names_path = rospy.get_param(
            "~class_names_path",
            osp.join(
                roslib.packages.get_pkg_dir("tam_object_detection"),
                "io/config/wrc_ycb_class_names.txt",
            ),
        )
        p_weight_path = rospy.get_param(
            "~weight_path",
            osp.join(
                roslib.packages.get_pkg_dir("tam_object_detection"),
                "io/models/yolov8-seg_wrc_ycb.engine",
            ),
        )
        self.p_device = rospy.get_param("~device", "cuda:0")
        self.p_confidence_th = rospy.get_param("~confidence_th", 0.3)
        self.p_iou_th = rospy.get_param("~iou_th", 0.65)
        self.p_max_area_raito = rospy.get_param("~max_area_ratio", 0.3)
        self.p_topk = rospy.get_param("~topk", 100)

        # p_camera_info_topic = rospy.get_param(
        #     "~camera_info_topic", "/camera/rgb/camera_info"
        # )
        p_rgb_topic = rospy.get_param("~rgb_topic", "/yamao/image_topic")
        self.p_use_depth = rospy.get_param("~use_depth", False)
        # p_depth_topic = rospy.get_param(
        #     "~depth_topic", "/camera/depth_registered/image_raw"
        # )

        self.p_use_segment = rospy.get_param("~use_segment", True)
        self.p_use_latest_image = rospy.get_param("~use_latest_image", False)
        self.p_get_pose = rospy.get_param("~get_pose", False)
        self.p_show_tf = rospy.get_param("~show_tf", False)
        self.p_max_distance = rospy.get_param("~max_distance", -1)
        self.p_specific_id = rospy.get_param("~specific_id", "")

        # Classes
        self.class_names = self.get_class_names(p_class_names_path)
        self.n_classes = len(self.class_names)
        self.colors = {
            cls: [random.randint(0, 255) for _ in range(3)]
            for i, cls in enumerate(self.class_names)
        }

        # TensorRT Engine
        self.setup_trt_engine(self.p_device, p_weight_path, self.p_use_segment)

        # Library
        self.bridge = CvBridge()
        self.tamtf = Transform()
        self.open3d = Open3D()

        # Publisher
        self.pub_register("result_image", "object_detection/image", Image)
        # self.pub_register("result_depth", "/object_depth/result", Image)
        # self.pub_register("result", f"{self.p_action_name}/detection", ObjectDetection)

        # Subscriber
        # self.camera_info = rospy.wait_for_message(p_camera_info_topic, CameraInfo)
        # self.set_camera_model(self.camera_info)
        self.camera_info = CameraInfo()
        self.camera_info.header.frame_id = "head_rgbd_sensor_rgb_frame"
        self.camera_info.height = 480
        self.camera_info.width = 640
        self.camera_info.distortion_model = "plumb_bob"
        self.set_camera_model(self.camera_info)

        self.msg_rgb = Image()
        if self.p_use_depth:
            self.msg_depth = CompressedImage()
            topics = {"msg_rgb": p_rgb_topic, "msg_depth": p_depth_topic}
            self.sync_sub_register("rgbd", topics, callback_func=self.subf_rgbd)
        else:
            self.sub_register("msg_rgb", p_rgb_topic, callback_func=self.subf_rgb)

    def set_camera_model(self, camera_info: CameraInfo) -> None:
        self.camera_model = PinholeCameraModel()
        self.camera_model.fromCameraInfo(camera_info)
        self.camera_frame_id = camera_info.header.frame_id

    def subf_rgb(self, rgb: CompressedImage) -> None:
        self.msg_rgb = rgb
        # self.cv_bgr = self.bridge.compressed_imgmsg_to_cv2(rgb)
        self.cv_bgr = self.bridge.imgmsg_to_cv2(rgb)
        self.set_update_ros_time("msg_rgb")

    def subf_rgbd(self, rgb: CompressedImage, depth: CompressedImage) -> None:
        self.msg_rgb = rgb
        self.msg_depth = depth
        self.cv_bgr = self.bridge.compressed_imgmsg_to_cv2(rgb)
        self.cv_depth = self.bridge.compressed_imgmsg_to_depth(depth)
        self.set_update_ros_time("msg_rgb")

    @staticmethod
    def get_class_names(path: str) -> List[str]:
        """ã‚¯ãƒ©ã‚¹åã‚’å–å¾—ã™ã‚‹

        Args:
            path (str): ã‚¯ãƒ©ã‚¹åãŒæ›¸ã‹ã‚ŒãŸãƒ•ã‚¡ã‚¤ãƒ«ï¼ˆ.txt)ï¼Ž

        Returns:
            List[str]: ã‚¯ãƒ©ã‚¹åãƒªã‚¹ãƒˆï¼Ž
        """
        with open(path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def setup_trt_engine(
        self, device: str, weight_path: str, use_segment: bool
    ) -> None:
        """TensorRTã‚’ã‚»ãƒƒãƒˆã‚¢ãƒƒãƒ—ã™ã‚‹

        Args:
            device (str): ãƒ‡ãƒã‚¤ã‚¹æƒ…å ±ï¼Ž
            weight_path (str): é‡ã¿ã®ãƒ‘ã‚¹ï¼Ž
            use_segment (bool): segmentationãƒ¢ãƒ‡ãƒ«ã‹ã©ã†ã‹ï¼Ž
        """
        device = torch.device(device)
        self.engine = TRTModule(weight_path, device)
        self.H, self.W = self.engine.inp_info[0].shape[-2:]
        if use_segment:
            self.engine.set_desired(["outputs", "proto"])
        else:
            self.engine.set_desired(["num_dets", "bboxes", "scores", "labels"])

    def inference(
        self, bgr: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """æŽ¨è«–

        Args:
            bgr (np.ndarray): å…¥åŠ›ç”»åƒï¼Ž

        Returns:
            Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]: æŽ¨è«–çµæžœï¼Ž
                bboxes, scores, labels, masksï¼ˆsegmentationã®å ´åˆã®ã¿ï¼‰
        """
        bgr, ratio, dwdh = yolov8_util.letterbox(bgr, (self.W, self.H))
        self.dw, self.dh = int(dwdh[0]), int(dwdh[1])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        tensor, self.seg_img = yolov8_util.blob(rgb, return_seg=True)
        dwdh = torch.asarray(dwdh * 2, dtype=torch.float32, device=self.p_device)
        tensor = torch.asarray(tensor, device=self.p_device)
        data = self.engine(tensor)

        if self.p_use_segment:
            bboxes, scores, labels, masks = yolov8_torch.seg_postprocess(
                data, bgr.shape[:2], self.p_confidence_th, self.p_iou_th
            )
        else:
            bboxes, scores, labels = yolov8_torch.det_postprocess(data)
            masks = None

        bboxes -= dwdh
        bboxes /= ratio

        return bboxes, scores, labels, masks

    def filter_elements_by_area_ratio(
        self,
        area_ratio: int,
        bboxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        masks: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        max_area = 640 * 480 * area_ratio
        indices = torch.tensor(
            [
                i
                for i, bbox in enumerate(bboxes)
                if (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) <= max_area
            ],
            device=self.p_device,
        )

        if len(indices) == 0:
            return (
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                None,
            )
        elif masks is None:
            return (
                bboxes[indices],
                scores[indices],
                labels[indices],
                None,
            )
        else:
            return (
                bboxes[indices],
                scores[indices],
                labels[indices],
                masks[indices],
            )

    def filter_elements_by_topk(
        self,
        topk: int,
        bboxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        masks: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        if len(scores) <= topk:
            return bboxes, scores, labels, masks

        return (
            bboxes[:topk],
            scores[:topk],
            labels[:topk],
            masks[:topk],
        )

    def filter_elements_by_id(
        self,
        ids: str,
        bboxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        masks: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]:
        """ç‰¹å®šãƒ©ãƒ™ãƒ«ã®çµæžœã®ã¿ã‚’æŠ½å‡ºã™ã‚‹

        Args:
            ids (str): ç‰¹å®šãƒ©ãƒ™ãƒ«ã®ãƒªã‚¹ãƒˆï¼ˆï¼ŒåŒºåˆ‡ã‚Šã®æ–‡å­—åˆ—ã§æŒ‡å®šï¼‰ï¼Ž
            bboxes (Tensor): BBoxæƒ…å ±ï¼Ž
            scores (Tensor): ã‚¹ã‚³ã‚¢æƒ…å ±ï¼Ž
            labels (Tensor): ãƒ©ãƒ™ãƒ«æƒ…å ±ï¼Ž
            masks (Optional[List[Tensor]]): ãƒžã‚¹ã‚¯æƒ…å ±ï¼Ž

        Returns:
            Tuple[Tensor, Tensor, Tensor, Optional[Tensor]]: æŠ½å‡ºçµæžœï¼Ž
        """
        ids_str = ids.split(",")
        ids_int = [int(x) for x in ids_str]
        indices = torch.tensor(
            [i for i, label in enumerate(labels) if int(label) in ids_int],
            device=self.p_device,
        )

        if len(indices) == 0:
            return (
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                None,
            )
        elif masks is None:
            return bboxes[indices], scores[indices], labels[indices], None
        else:
            return (
                bboxes[indices],
                scores[indices],
                labels[indices],
                masks[indices],
            )

    def filter_elements_by_pose(
        self,
        poses: List[Pose],
        bboxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        masks: Optional[Tensor],
    ) -> Tuple[List[Pose], Tensor, Tensor, Tensor, Optional[Tensor]]:
        """åº§æ¨™ã®ã‚ã‚‹çµæžœã®ã¿ã‚’æŠ½å‡ºã™ã‚‹

        Args:
            poses (List[Pose]): åº§æ¨™ãƒªã‚¹ãƒˆï¼Ž
            bboxes (Tensor): BBoxæƒ…å ±ï¼Ž
            scores (Tensor): ã‚¹ã‚³ã‚¢æƒ…å ±ï¼Ž
            labels (Tensor): ãƒ©ãƒ™ãƒ«æƒ…å ±ï¼Ž
            masks (Optional[Tensor]): ãƒžã‚¹ã‚¯æƒ…å ±ï¼Ž

        Returns:
            Tuple[List[Pose], Tensor, Tensor, Tensor, Optional[Tensor]]: æŠ½å‡ºçµæžœï¼Ž
        """
        indices = [i for i, x in enumerate(poses) if x is not None]
        indices_tensor = torch.tensor(indices, device=self.p_device)
        if len(indices) == 0:
            return (
                [],
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                None,
            )
        elif masks is None:
            return (
                np.array(poses)[indices].tolist(),
                bboxes[indices_tensor],
                scores[indices_tensor],
                labels[indices_tensor],
                None,
            )
        else:
            return (
                np.array(poses)[indices].tolist(),
                bboxes[indices_tensor],
                scores[indices_tensor],
                labels[indices_tensor],
                masks[indices_tensor],
            )

    # def visualize(
    #     self,
    #     image: np.ndarray,
    #     bboxes: Tensor,
    #     labels: Tensor,
    #     masks: Optional[Tensor],
    # ) -> np.ndarray:
    #     """çµæžœã®å¯è¦–åŒ–

    #     Args:
    #         image (np.ndarray): å…¥åŠ›ç”»åƒï¼Ž
    #         bboxes (Tensor): BBoxæƒ…å ±ï¼Ž
    #         labels (Tensor): ãƒ©ãƒ™ãƒ«æƒ…å ±ï¼Ž
    #         masks (Optional[List[Tensor]], optional): ãƒžã‚¹ã‚¯æƒ…å ±. Defaults to None.

    #     Returns:
    #         np.ndarray: æç”»ç”»åƒï¼Ž
    #     """
    #     result_image = image.copy()

    #     if masks is not None:
    #         self.seg_img = torch.asarray(
    #             self.seg_img[
    #                 self.dh : self.H - self.dh, self.dw : self.W - self.dw, [2, 1, 0]
    #             ],
    #             device=self.p_device,
    #         )
    #         self.mask = masks[:, self.dh : self.H - self.dh, self.dw : self.W - self.dw, :]
    #         indices = (labels % len(yolov8_msk_clr)).long()
    #         mask_colors = torch.asarray(yolov8_msk_clr, device=self.p_device)[indices]
    #         mask_colors = mask_colors.view(-1, 1, 1, 3) * yolov8_alp
    #         mask_colors = self.mask @ mask_colors
    #         inv_alph_masks = (1 - self.mask * 0.5).cumprod(0)
    #         mcs = (mask_colors * inv_alph_masks).sum(0) * 2
    #         self.seg_img = (self.seg_img * inv_alph_masks[-1] + mcs) * 255
    #         result_image = cv2.resize(
    #             self.seg_img.cpu().numpy().astype(np.uint8),
    #             result_image.shape[:2][::-1],
    #         )

    #     for (bbox, label) in zip(bboxes, labels):
    #         bbox = bbox.round().int().tolist()
    #         cls = self.class_names[int(label)]
    #         color = self.colors[cls]
    #         cv2.rectangle(result_image, bbox[:2], bbox[2:], color, 2)
    #         cv2.putText(
    #             result_image,
    #             f"{label}:{cls}",
    #             (bbox[0], bbox[1] - 2),
    #             cv2.FONT_HERSHEY_SIMPLEX,
    #             1.0,
    #             [0, 0, 0],
    #             # [255, 255, 255],
    #             thickness=2,
    #         )

    #     return result_image

    def visualize(
        self,
        image: np.ndarray,
        bboxes: Tensor,
        labels: Tensor,
        masks: Optional[Tensor],
    ) -> np.ndarray:
        """çµæžœã®å¯è¦–åŒ–
        Args:
            image (np.ndarray): å…¥åŠ›ç”»åƒï¼Ž
            bboxes (Tensor): BBoxæƒ…å ±ï¼Ž
            labels (Tensor): ãƒ©ãƒ™ãƒ«æƒ…å ±ï¼Ž
            masks (Optional[List[Tensor]], optional): ãƒžã‚¹ã‚¯æƒ…å ±. Defaults to None.
        Returns:
            np.ndarray: æç”»ç”»åƒï¼Ž
        """
        result_image = image.copy()
        if masks is not None:
            self.seg_img = torch.asarray(
                self.seg_img[
                    self.dh : self.H - self.dh, self.dw : self.W - self.dw, [2, 1, 0]
                ],
                device=self.p_device,
            )
            self.mask = masks[:, self.dh : self.H - self.dh, self.dw : self.W - self.dw, :]
            indices = (labels % len(yolov8_msk_clr)).long()
            mask_colors = torch.asarray(yolov8_msk_clr, device=self.p_device)[indices]
            mask_colors = mask_colors.view(-1, 1, 1, 3) * yolov8_alp
            mask_colors = self.mask @ mask_colors
            inv_alph_masks = (1 - self.mask * 0.5).cumprod(0)
            mcs = (mask_colors * inv_alph_masks).sum(0) * 2
            # self.seg_img = (self.seg_img * inv_alph_masks[-1] + mcs) * 255
            # result_image = cv2.resize(
            #     self.seg_img.cpu().numpy().astype(np.uint8),
            #     result_image.shape[:2][::-1],
            # )
        som_number = 1 ##############################################yamaotuika
        result_name = ""
        for (bbox, label) in zip(bboxes, labels):
            bbox = bbox.round().int().tolist()
            cls = self.class_names[int(label)]
            color = self.colors[cls]
            cv2.rectangle(result_image, bbox[:2], bbox[2:], color, 2)
            if som_number >= 10:
                # cv2.rectangle(result_image, (bbox[0]-2, bbox[1]-20), (bbox[0]+25, bbox[1]), [255, 255, 255], thickness=cv2.FILLED)#
                cv2.rectangle(result_image, (int(((bbox[0]+bbox[2]) / 2) - 13), int(((bbox[1]+bbox[3]) / 2) - 10)), (int(((bbox[0]+bbox[2]) / 2) + 13), int(((bbox[1]+bbox[3]) / 2) + 10)), [0, 0, 0], thickness=cv2.FILLED)#
                cv2.putText(#
                    result_image,
                    f"{som_number}",
                    (int((bbox[0]+bbox[2])/2-13), int((bbox[1]+bbox[3])/2+7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    # [0, 0, 0],
                    [255, 255, 255],
                    thickness=2,
                )
            else:
                # cv2.rectangle(result_image, (bbox[0]-2, bbox[1]-20), (bbox[0]+15, bbox[1]), [255, 255, 255], thickness=cv2.FILLED)#
                cv2.rectangle(result_image, (int(((bbox[0]+bbox[2]) / 2) - 10), int(((bbox[1]+bbox[3]) / 2) - 10)), (int(((bbox[0]+bbox[2]) / 2) + 10), int(((bbox[1]+bbox[3]) / 2) + 10)), [0, 0, 0], thickness=cv2.FILLED)#
                cv2.putText(#
                    result_image,
                    f"{som_number}",
                    (int((bbox[0]+bbox[2])/2-7), int((bbox[1]+bbox[3])/2+7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    # [0, 0, 0],
                    [255, 255, 255],
                    thickness=2,
                )
            result_name = result_name + str(som_number) + "." + str(cls) + ", "
            som_number += 1
            
        rospy.set_param("/result_name", result_name)
            # cv2.putText(
            #     result_image,
            #     f"{label}:{cls}",
            #     (bbox[0], bbox[1] - 2),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1.0,
            #     [0, 0, 0],
            #     # [255, 255, 255],
            #     thickness=2,
            # )
        return result_image


    def get_3d_poses(self, depth: np.ndarray, bboxes: Tensor) -> List[Optional[Pose]]:
        """ã‚«ãƒ¡ãƒ©åº§æ¨™ç³»ã§ã®ä¸‰æ¬¡å…ƒåº§æ¨™ã‚’å–å¾—ã™ã‚‹ï¼ˆç‰©ä½“ä¸­å¿ƒï¼‰

        Args:
            depth (np.ndarray): Depthç”»åƒï¼Ž
            bboxes (Tensor): BBoxæƒ…å ±ï¼Ž

        Returns:
            List[Optional[Pose]]: å„ã‚ªãƒ–ã‚¸ã‚§ã‚¯ãƒˆã®3æ¬¡å…ƒåº§æ¨™ï¼Ž
                è¨ˆç®—ã§ããªã‹ã£ãŸå ´åˆï¼ŒNoneãŒæ ¼ç´ã•ã‚Œã‚‹ï¼Ž
        """
        poses: List[Optional[Pose]] = []
        for box in bboxes:
            w, h = box[2] - box[0], box[3] - box[1]
            cx, cy = int(box[0] + w / 2), int(box[1] + h / 2)
            crop_depth = depth[cy - 2 : cy + 3, cx - 2 : cx + 3] * 0.001
            flat_depth = crop_depth[crop_depth != 0].flatten()
            if len(flat_depth) == 0:
                poses.append(None)
                continue
            mean_depth = np.mean(flat_depth)
            uv = list(self.camera_model.projectPixelTo3dRay((cx, cy)))
            uv[:] = [x / uv[2] for x in uv]
            uv[:] = [x * mean_depth for x in uv]
            if self.p_max_distance < 0 or (
                self.p_max_distance > 0 and self.p_max_distance > uv[2]
            ):
                poses.append(Pose(Point(*uv), Quaternion(0, 0, 0, 1)))
            else:
                poses.append(None)
        return poses

    def show_tf(self, poses: List[Optional[Pose]], labels: Tensor) -> None:
        """TFã‚’é…ä¿¡ã™ã‚‹

        Args:
            poses (List[Optional[Pose]]): 3æ¬¡å…ƒåº§æ¨™ãƒªã‚¹ãƒˆï¼Ž
            labels (Tensor): ãƒ©ãƒ™ãƒ«æƒ…å ±ï¼Ž
        """
        label_dict: Dict[str, int] = {}
        for pose, label in zip(poses, labels):
            label = int(label)
            if pose is None:
                continue
            if label in label_dict:
                label_dict[label] += 1
            else:
                label_dict[label] = 1
            self.tamtf.send_transform(
                f"{self.class_names[int(label)]}_{label_dict[label]}",
                self.camera_frame_id,
                pose,
            )

    def create_segment_msg(self, num: int) -> CompressedImage:
        """segmentãƒ¡ãƒƒã‚»ãƒ¼ã‚¸ã‚’ä½œæˆã™ã‚‹

        Args:
            num (int): æ¤œå‡ºæ•°ï¼Ž

        Returns:
            List[CompressedImage]: segmentãƒ¡ãƒƒã‚»ãƒ¼ã‚¸ãƒªã‚¹ãƒˆï¼Ž
        """
        mask = self.mask.to("cpu").detach().numpy().copy() * 255
        mask_split = np.split(mask, num)
        seg_msgs = []
        for i in range(num):
            m = mask_split[i].reshape((480, 640))
            seg_msg = self.bridge.cv2_to_compressed_imgmsg(m)
            seg_msgs.append(seg_msg)
        return seg_msgs

    def create_object_detection_msg(
        self,
        msg_rgb: CompressedImage,
        bboxes: Tensor,
        scores: Tensor,
        labels: Tensor,
        msg_depth: Optional[np.ndarray] = None,
        poses: Optional[List] = None,
    ) -> ObjectDetection:
        """ObjectDetection.msgã‚’ä½œæˆã™ã‚‹

        Args:
            msg_rgb (CompressedImage): RGBãƒ¡ãƒƒã‚»ãƒ¼ã‚¸ï¼Ž
            bboxes (Tensor): BBoxæƒ…å ±ï¼Ž
            scores (Tensor): ã‚¹ã‚³ã‚¢æƒ…å ±ï¼Ž
            labels (Tensor): ãƒ©ãƒ™ãƒ«æƒ…å ±ï¼Ž
            msg_depth (Optional[CompressedImage, optional): Depthãƒ¡ãƒƒã‚»ãƒ¼ã‚¸. Defaults to None.
            poses (Optional[List], optional): 3æ¬¡å…ƒåº§æ¨™ãƒªã‚¹ãƒˆ. Defaults to None.

        Returns:
            ObjectDetection: ãƒ¡ãƒƒã‚»ãƒ¼ã‚¸ï¼Ž
        """
        msg = ObjectDetection()
        msg.header = Header()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = self.camera_frame_id
        msg.is_detected = True
        # msg.camera_info = self.camera_info
        msg.rgb = msg_rgb
        if self.p_use_segment:
            msg.segments = self.create_segment_msg(len(bboxes))
        for bbox, score, label in zip(bboxes, scores, labels):
            msg.bbox.append(BBox())
            msg.bbox[-1].id = int(label)
            msg.bbox[-1].name = self.class_names[int(label)]
            msg.bbox[-1].score = float(score)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y = bbox[0] + w / 2, bbox[1] + h / 2
            msg.bbox[-1].x = int(x)
            msg.bbox[-1].y = int(y)
            msg.bbox[-1].w = int(w)
            msg.bbox[-1].h = int(h)
        if msg_depth is not None and poses is not None:
            msg.depth = msg_depth
            for pose in poses:
                msg.pose.append(Pose3D())
                if pose is None:
                    msg.pose[-1].is_valid = False
                else:
                    msg.pose[-1].is_valid = True
                    msg.pose[-1].frame_id = self.camera_frame_id
                    msg.pose[-1].position = pose.position
                    msg.pose[-1].orientation = pose.orientation
        return msg

    def pub_no_objects(self, cv_bgr: np.ndarray) -> None:
        """çµæžœãªã—ã®Publish

        Args:
            cv_bgr (np.ndarray): RGBç”»åƒï¼Ž
        """
        msg = ObjectDetection()
        msg.is_detected = False
        self.pub.result.publish(msg)
        msg_img = self.bridge.cv2_to_imgmsg(cv_bgr)
        self.pub.result_image.publish(msg_img)

    def run(self) -> None:
        if self.run_enable is False:
            return

        # RGBç”»åƒå–å¾—
        if not hasattr(self, "cv_bgr"):
            return
        if self.p_use_latest_image:
            self.wait_for_message("msg_rgb")
        cv_bgr = self.cv_bgr
        msg_rgb = self.msg_rgb

        # æŽ¨è«–
        try:
            bboxes, scores, labels, masks = self.inference(cv_bgr)
        except Exception:
            self.logwarn("No objects detected.")
            self.pub_no_objects(cv_bgr)
            return

        # size validation
        bboxes, scores, labels, masks = self.filter_elements_by_area_ratio(
            self.p_max_area_raito, bboxes, scores, labels, masks
        )

        # top k
        bboxes, scores, labels, masks = self.filter_elements_by_topk(
            self.p_topk, bboxes, scores, labels, masks
        )

        # ç‰¹å®šãƒ©ãƒ™ãƒ«ã®ã¿æŠ½å‡º
        if self.p_specific_id != "":
            bboxes, scores, labels, masks = self.filter_elements_by_id(
                self.p_specific_id, bboxes, scores, labels, masks
            )
        if len(bboxes) == 0:
            self.pub_no_objects(cv_bgr)
            return

        # Depthç”»åƒå‡¦ç†
        if self.p_use_depth:
            if not hasattr(self, "cv_depth"):
                return
            cv_depth = self.cv_depth
            msg_depth = self.msg_depth

            if cv_depth is None:
                return

            # 3æ¬¡å…ƒåº§æ¨™ã®å–å¾—
            poses = self.get_3d_poses(cv_depth, bboxes)

            # åº§æ¨™ã®ç„¡ã„çµæžœã®å‰Šé™¤
            poses, bboxes, scores, labels, masks = self.filter_elements_by_pose(
                poses, bboxes, scores, labels, masks
            )
            if len(bboxes) == 0:
                self.pub_no_objects(cv_bgr)
                return

            # TFé…ä¿¡
            if self.p_show_tf and poses != []:
                self.show_tf(poses, labels)
        else:
            msg_depth = None
            poses = None

        # å¯è¦–åŒ–
        cv_result = self.visualize(cv_bgr, bboxes, labels, masks)
        msg_result = self.bridge.cv2_to_imgmsg(cv_result)
        self.pub.result_image.publish(msg_result)

        # æŽ¨è«–çµæžœ
        # result = self.create_object_detection_msg(
        #     msg_rgb, bboxes, scores, labels, msg_depth, poses
        # )
        # self.pub.result.publish(bboxes)

        #############################################yamaotuika

        # å¯è¦–åŒ–
        # yamao_result = self.visualize(cv_depth, bboxes, labels, masks)
        # msg_result = self.bridge.cv2_to_imgmsg(yamao_result)
        # self.pub.result_depth.publish(msg_result)


def main():
    rospy.init_node(os.path.basename(__file__).split(".")[0])

    p_loop_rate = rospy.get_param(rospy.get_name() + "/loop_rate", 30)
    loop_wait = rospy.Rate(p_loop_rate)

    cls = YOLOv8TensorRT()
    rospy.on_shutdown(cls.delete)
    while not rospy.is_shutdown():
        try:
            cls.run()
        except rospy.exceptions.ROSException as e:
            rospy.logerr(f"[{rospy.get_name()}]: FAILURE")
            rospy.logerr(e)
        loop_wait.sleep()


if __name__ == "__main__":
    main()