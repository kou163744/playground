#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import cv2
import copy
import rospy
import smach
import roslib
import openai
import base64
import sqlite3
import numpy as np
from tamlib.utils import Logger
from std_srvs.srv import SetBool, SetBoolRequest
from cv_bridge import CvBridge, CvBridgeError
sys.path.append(roslib.packages.get_pkg_dir("interactive_customer_service") + "/script")
from parser_utils import Utils

from interactive_customer_service.msg import Conversation
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import Image


class Interaction(smach.State, Logger):
    def __init__(self, outcomes):
        smach.State.__init__(self, outcomes=outcomes)
        Logger.__init__(self, loglevel="DEBUG")
        self.ics_utils = Utils()

        package_name = "interactive_customer_service"
        io_dir = f"{roslib.packages.get_pkg_dir(package_name)}/io/"
        self.img_extension = "png"
        self.image_path = os.path.join(io_dir, "customer_img.png")
        database_dir = f"{roslib.packages.get_pkg_dir(package_name)}/io/database/"
        database_path = os.path.join(database_dir, "icra_2024.db")

        self.loginfo(database_dir)
        self.loginfo(database_path)

        # データベースに接続（データベースがない場合は新たに作成される）
        self.conn = sqlite3.connect(database_path)
        self.query = "SELECT * FROM items"

        # カーソルオブジェクトを作成
        self.cursor = self.conn.cursor()

        self.api_key = os.environ.get("OPENAI_API")
        self.gpt_version = "gpt-4"
        self.gpt_vision_version = "gpt-4-vision-preview"

        if self.api_key is None:
            self.logerr("OPENAI_API is not set.")
            sys.exit()
        else:
            openai.api_key = self.api_key

        self.ics_prompt_orig = [
            {"role": "system", "content": "あなたはコンビニエンスストアで働いているサービスロボットです．"},
            {"role": "system", "content": "お客さんがある商品を求めています．その商品のメインカテゴリ，サブカテゴリを順番に推定し，最終的に必要としている商品名を突き止めてください．"},
            # {"role": "system", "content": "商品名の末尾についている-1000や-2000といった数字は無視して考えてください．"},
            {"role": "system", "content": "メインカテゴリには以下のものがあります．"},
            {"role": "system", "content": "わさびや醤油などの調味料を含む[seasoning]"},
            {"role": "system", "content": "カップ麺やカップ焼きそばなどの[instant_noodles]"},
            {"role": "system", "content": "ペットボトル飲料，のみものを示す[bevarage]"},
            {"role": "system", "content": "チュッパチャップスやのど飴，チョコ，ポテトチップス，ガムなどを示す[sweets]"},
            {"role": "system", "content": "カレー，シチュー，インスタントスープのもととなるものを示す[roux]"},
            {"role": "system", "content": "洗剤，ティッシュ，ハンドクリームなどを示す[daily_necessities]"},
            {"role": "system", "content": "リストから選択する際に，適したものがどれかわからなかった場合は，unknownと出力してください．"},
            {"role": "system", "content": "リストの選択肢が1つしかない場合は，その選択肢をそのまま出力してください．"},
            {"role": "system", "content": "以下は例です．"},
            {"role": "user", "content": "要望文に最も合うメインカテゴリを以下のリストから選択し，メインカテゴリのみを出力してください．リスト[seasoning, instant_noodles, bevarages, sweets, roux, daily_necessities], 要望文: Please give me a garlic seasoning."},
            {"role": "assistant", "content": "seasoning"},
            {"role": "user", "content": "要望文に最も合うサブカテゴリを以下のリストから選択し，サブカテゴリのみを出力してください．リスト[mixed_seasoning, spicy, garlic, wasabi, ginger, soysauce, other], 要望文: Please give me a garlic seasoning."},
            {"role": "assistant", "content": "garlic"},
            {"role": "user", "content": "要望文に最も合う商品名を選択し，商品名のみを出力してください．リスト[7i_garlic-1000], 要望文: Please give me a garlic seasoning."},
            {"role": "assistant", "content": "7i_garlic-1000"},
            {"role": "system", "content": "以下より，新たに推論を開始してください．"},
        ]

        # 制御用変数
        self.DB_ID = 0
        self.DB_NAME = 1
        self.DB_TYPE = 2
        self.DB_SUBTYPE = 3

        self.customer_img = None

        # ros interface
        self.bridge = CvBridge()
        rospy.Subscriber("/interactive_customer_service/customer_image", Image, self.cb_customer_img)

    def cb_customer_img(self, msg: Image) -> None:
        """お客さんからの画像を受け取るコールバック関数
        """
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.customer_img = cv_img
        self.with_image = True

    def _reset_query(self) -> None:
        """検索分をリセットする関数
        """
        self.query = "SELECT * FROM items"

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def extract_category(self, input_string: str):
        # 角括弧 [] 内の文字を抽出
        category = input_string[input_string.find('[') + 1 : input_string.find(']')]

        # 丸括弧 () 内の文字を抽出
        sub_category = input_string[input_string.find('(') + 1 : input_string.find(')')]

        return category, sub_category

    def get_main_category(self, instruction_txt: str, cv_img: np.ndarray = None) -> str:
        """カスタマーの指示文から対象カテゴリを推定する関数
        Args:
            instruction_txt(str): カスタマーからの文章
            cv_img(np.ndarray): カスタマーからの画像
                defaults set to None
        Returns:
            str
        """
        openai.api_key = self.api_key

        # 文章のみの指示の場合
        if cv_img is None:
            self.ics_prompt.append({"role": "user", "content": instruction_txt})
            response = openai.ChatCompletion.create(
                model=self.gpt_version,
                messages=self.ics_prompt
            )

            try:
                category, sub_category = self.extract_category(response['choices'][0]['message']['content'])
                return category, sub_category
            except Exception as e:
                self.logwarn(e)

        else:
            # openaiで扱える画像形式に変換
            cv2.imwrite(self.image_path, cv_img)
            self.image_base64 = self.encode_image(image_path=self.image_path)
            self.url = f"data:image/{self.img_extension};base64,{self.image_base64}"

            self.ics_prompt.append(
                {
                    "role": "user", "content":
                    [
                        {
                            "type": "text",
                            "text": instruction_txt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": self.url
                            }
                        }
                    ]
                }
            )

            try:
                category, sub_category = self.extract_category(response['choices'][0]['message']['content'])
                return category, sub_category
            except Exception as e:
                self.logwarn(e)

    def create_question(self, rows) -> str:
        """現在ヒットしている商品一覧とクエリをもとに，次の質問文を作成する関数
        Args:
            rows(): 現在のクエリから導かれた検索結果
        """

    # def execute_with_query(self):
    #     # 大カテゴリと小カテゴリを推測する
    #     category, sub_category = self.get_main_category(instruction_txt)
    #     self.logdebug(f"main category: {category}, sub category: {sub_category}")
    #     self.query = f"SELECT * FROM items WHERE type LIKE \'%{category}%\'"
    #     self.logdebug(f"query is: {self.query}")

    #     while not rospy.is_shutdown():
    #         self.cursor.execute(self.query)
    #         rows = self.cursor.fetchall()

    #         if len(rows) == 1:
    #             # 商品がひとつだけになったらクエリをリセットして商品を取りに行く
    #             target_item_name = rows[0][self.DB_NAME]
    #             target_item_id = rows[0][self.DB_ID]

    #             self.loginfo(target_item_name)
    #             self.loginfo(target_item_id)

    #             self._reset_query()
    #             break
    #         elif len(rows) <= 0:
    #             # 検索結果がなにもなかった場合
    #             self.loginfo("検索の結果，なにもヒットしなくなりました．")
    #             self._reset_query()
    #             return "except"
    #         else:
    #             # 更に絞り込むための質問を作成する
    #             self.logdebug("============== 現時点で絞り込めている商品一覧 ==============")
    #             for row in rows:
    #                 self.logdebug(row)
    #             self.logdebug("======================================================")

    #     # データベース接続を閉じる
    #     self.conn.close()

    def execute(self, userdata):
        # パラメータの初期化
        self.with_image = False
        self.customer_img = None
        rospy.set_param("/interactive_customer_service/orderd_id", 999)

        # お客さんからのメッセージを取得する
        while not rospy.is_shutdown():
            try:
                customer_msg = rospy.wait_for_message("/interactive_customer_service/message/customer", Conversation, timeout=10)
                if customer_msg.type == "customer_message":
                    instruction_txt = customer_msg.detail
                    break
                else:
                    continue
            except Exception as e:
                self.logwarn(e)
                return "except"

        # customer imageが送信される可能性を考慮して，処理を停止する
        rospy.sleep(5)

        self.loginfo(instruction_txt)
        self.loginfo(self.with_image)

        target_item = self.ics_utils.estimate_customer_order(instruction_txt=instruction_txt, cv_img=self.customer_img)
        target_id = self.ics_utils.get_item_index(target_item=target_item)

        if target_id is False:
            return "except"
        else:
            self.loginfo(target_id)

        rospy.set_param("/interactive_customer_service/orderd_item", target_item)
        rospy.set_param("/interactive_customer_service/orderd_id", target_id)

        return "next"
