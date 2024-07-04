#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from tam_object_detection.msg import ObjectDetection
import cv2
import numpy as np
import yaml
import sys
class ImageAndRosbagSaver:
    def __init__(self, image_topics, save_path):
        self.args = sys.argv
        self.image_topics = image_topics
        self.save_path = save_path
        self.bridge = CvBridge()
        self.image_subscribers = {}
        self.topic = ""
        self.image_subscribers[0] = rospy.Subscriber("/object_detection/image", Image, self.image_callback)
        # self.image_subscribers[image_topics[0]] = rospy.Subscriber(image_topics[0], Image, self.detection_callback)
        # self.image_subscribers[image_topics[1]] = rospy.Subscriber(image_topics[1], Image, self.image_callback)
        # self.image_subscribers[image_topics[2]] = rospy.Subscriber(image_topics[2], Image, self.depth_callback)
        # self.image_subscribers[image_topics[3]] = rospy.Subscriber(image_topics[3], Image, self.detection_depth_callback)
        # self.image_subscribers["/object_detection/detection"] = rospy.Subscriber("/object_detection/detection", ObjectDetection, self.detection_data_callback)
    
    def detection_data_callback(self, msg):
        try:
            # 画像がdepthであれば，正規化する
            f = open('jikken1_tuika/result.txt', 'w')
            f.write(str(msg.bbox))
            f.close
        except CvBridgeError as e:
            rospy.logerr(e)
    
    # def image_callback(self, msg):
    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    #         # 画像をjpg形式で保存
    #         image_filename = self.save_path + "/" + "rgb_input.jpg"
    #         cv2.imwrite(image_filename, cv_image)
    #         rospy.loginfo(f"Image saved as {image_filename}")
    #     except CvBridgeError as e:
    #         rospy.logerr(e)
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "8UC3")
                
            print("callback")
            # 画像をjpg形式で保存
            image_filename = self.save_path + "/" + f"rgb_input{self.args[1]}.jpg"
            cv2.imwrite(image_filename, cv_image)
            rospy.loginfo(f"Image saved as {image_filename}")
            result = rospy.get_param("/result_name")
            file_path = "../io/data/level2_sii.yaml"
            with open(file_path, 'r') as file:
                data = yaml.safe_load(file)

            # Add the new data to the existing data
            data.update({f"image{self.args[1]}": f"{result}"})

            # Write the updated data back to the YAML file
            with open(file_path, 'w') as file:
                yaml.safe_dump(data, file, default_flow_style=False)
        except CvBridgeError as e:
            rospy.logerr(e)
            
    def depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            pixels = cv_image.astype(float)
            pixels = np.where(pixels == 0, 500, pixels)
            pixels = np.where(pixels >= 1500, 500, pixels)
            amin=np.amin(pixels)
            amax=np.amax(pixels)
            scale = 255.0/(amax-amin) # 256.0/(amax-amin+1)
            pixelsByte = np.copy(pixels)
            pixelsByte = pixelsByte*scale # 0-255の範囲にリスケール
            pixelsByte = np.uint8(pixelsByte) # 符号なしバイトに変換
            cv_image = pixelsByte
            
            # 画像をjpg形式で保存
            image_filename = self.save_path + "/" + "depth_input.jpg"
            cv2.imwrite(image_filename, cv_image)
            rospy.loginfo(f"Image saved as {image_filename}")
        except CvBridgeError as e:
            rospy.logerr(e)
    
    def detection_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            
            # 画像をjpg形式で保存
            image_filename = self.save_path + "/" + "rgb_result.jpg"
            cv2.imwrite(image_filename, cv_image)
            rospy.loginfo(f"Image saved as {image_filename}")
        except CvBridgeError as e:
            rospy.logerr(e)
            
    def detection_depth_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg)
            # pixels = cv_image.astype(float)
            # pixels = np.where(pixels == 0, 500, pixels)
            # pixels = np.where(pixels >= 1500, 500, pixels)
            # amin=np.amin(pixels)
            # amax=np.amax(pixels)
            # scale = 255.0/(amax-amin) # 256.0/(amax-amin+1)
            # pixelsByte = np.copy(pixels)
            # pixelsByte = pixelsByte*scale # 0-255の範囲にリスケール
            # pixelsByte = np.uint8(pixelsByte) # 符号なしバイトに変換
            # cv_image = pixelsByte
            
            # 画像をjpg形式で保存
            image_filename = self.save_path + "/" + "depth_result.jpg"
            cv2.imwrite(image_filename, cv_image)
            rospy.loginfo(f"Image saved as {image_filename}")
        except CvBridgeError as e:
            rospy.logerr(e)

def main():
    rospy.init_node('image_and_rosbag_saver', anonymous=True)
    # 保存先ディレクトリのパスを指定
    save_path = "../io/data/level2_results_sii"
    # 受信したいImageトピックを指定
    image_topics = ["/object_detection/image", "/hsrb/head_rgbd_sensor/rgb/image_raw", "/hsrb/head_rgbd_sensor/depth_registered/image_raw", "/object_depth/result"]
    # Rosbagファイルの名前を指定
    ImageAndRosbagSaver(image_topics, save_path)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down...")
if __name__ == '__main__':
    main()
