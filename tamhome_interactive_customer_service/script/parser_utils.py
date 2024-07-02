#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import copy
# import rospy
import random
import base64
from openai import OpenAI

###ros関係はコメントアウト
# import roslib
import sqlite3
import numpy as np
# from tamlib.utils import Logger
from items import ConvinienceItems
import logging

# from interactive_customer_service.msg import Conversation

import argparse


class Utils(ConvinienceItems):
    def __init__(self):
        logging.basicConfig(level=logging.DEBUG)
        # Logger.__init__(self, loglevel="INFO")
        ConvinienceItems.__init__(self)

        self.package_name = "interactive_customer_service"

        parser = argparse.ArgumentParser()
        parser.add_argument('--llama', action = 'store_true')
        args = parser.parse_args()

        self.api_key = os.environ.get("OPENAI_API")
        if args.llama:
            self.client = OpenAI(base_url="http://localhost:8000/v1", api_key = self.api_key)
        else:
            self.client = OpenAI(api_key = self.api_key)
        self.gpt_version = "gpt-4"
        self.gpt_vision_version = "gpt-4-vision-preview"
        self.gpt_vision_version = "gpt-4o"

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
            # {"role": "system", "content": "選択の際の確信度を高・中・低の3段階で分類し，低のときのみ出力の末尾に?を追加してください．"},
            {"role": "system", "content": "以下は例です．"},
            {"role": "user", "content": "要望文に最も合うメインカテゴリを以下のリストから選択し，メインカテゴリのみを出力してください．リスト[seasoning, instant_noodles, bevarages, sweets, roux, daily_necessities], 要望文: Please give me a garlic seasoning."},
            {"role": "assistant", "content": "seasoning"},
            {"role": "user", "content": "要望文に最も合うサブカテゴリを以下のリストから選択し，サブカテゴリのみを出力してください．リスト[mixed_seasoning, spicy, garlic, wasabi, ginger, soysauce, other], 要望文: Please give me a garlic seasoning."},
            {"role": "assistant", "content": "garlic"},
            {"role": "user", "content": "要望文に最も合う商品名を選択し，商品名のみを出力してください．リスト[7i_garlic-1000], 要望文: Please give me a garlic seasoning."},
            {"role": "assistant", "content": "7i_garlic-1000"},

            # {"role": "user", "content": "要望文に最も合うメインカテゴリを以下のリストから選択し，メインカテゴリのみを出力してください．リスト[seasoning, instant_noodles, bevarages, sweets, roux, daily_necessities], 要望文: Please bring me the xylitol chewing gum."},
            # {"role": "assistant", "content": "sweets"},
            # {"role": "user", "content": "要望文に最も合うサブカテゴリを以下のリストから選択し，サブカテゴリのみを出力してください．リスト[throat_lozenge, chupachups, cookie, chewing_gum], 要望文: Please bring me the xylitol chewing gum."},
            # {"role": "assistant", "content": "chewing_gum_xylitol"},
            # {"role": "user", "content": "要望文に最も合う商品名を選択し，商品名のみを出力してください．リスト[xylitol_freshmint-1000, 11_xylitol-1000, xylitol_peach-1000, xylitol_white_pink-1000, xylitol_uruoi-3000], 要望文: Please bring me the xylitol chewing gum."},
            # {"role": "assistant", "content": "xylitol_freshmint-1000?"},
            # {"role": "user", "content": "要望文に最も合う商品名を選択し，商品名のみを出力してください．リスト[xylitol_freshmint-1000, xylitol_peach-1000, xylitol_white_pink-1000, xylitol_uruoi-3000], 要望文: Please bring me the xylitol chewing gum."},
            # {"role": "assistant", "content": "11_xylitol-1000"},

            {"role": "system", "content": "以下より，新たに推論を開始してください．"},
        ]

        self.make_question_prompt = [
            {"role": "system", "content": "あなたはコンビニエンスストアで働いているサービスロボットです．"},
            {"role": "system", "content": "お客さんが求めている商品を特定するために必要な質問文を作成してください．"},
            {"role": "system", "content": "プロンプトには，その商品についてすでにわかっている項目と，特定したい項目が与えられます．これは，商品の味や形状などのカテゴリ情報です．"},
            {"role": "system", "content": "質問は必ずYes, Noで答えられるクローズドクエスチョンにしてください．"},
            {"role": "user", "content": "seasoning, spicy"},
            {"role": "assistant", "content": "Is the seasoning you are looking for spicy?"},
            {"role": "user", "content": "sweets, cylindrical-shaped"},
            {"role": "assistant", "content": "Is the shape of the snacks you are looking for cylindrical?"},
            {"role": "user", "content": "bevarage, coke"},
            {"role": "assistant", "content": "Is the bevarage you are looking for coke?"},
        ]

        # self.pub_question = rospy.Publisher("/interactive_customer_service/message/robot", Conversation, queue_size=10)

    def get_item_index(self, target_item) -> int:
        """対象商品のIDを返す関数
        Args:
            target_item(str): 対象商品
        Return
            int: 商品のID，見つからなかった場合はFalse
        """
        try:
            target_id = self.id_list.index(target_item) + 1
            return target_id
        except ValueError:
            logging.warn(f"{target_item} is not in the list")
            return False

    def make_question(self, main_category, target) -> str:
        """わかっている情報と特定したい項目をもとにした質問文を生成する
        """
        self.make_question_prompt.append({"role": "user", "content": f"{main_category}, {target}"})
        response = self.client.chat.completions.create(
            model=self.gpt_vision_version,
            messages=self.ics_prompt
        )

        question_str = response.choices[0].message.content

        return question_str

    def ask_question(self, question_str: str, with_ros=True) -> str:
        """質問をする関数
        Args:
            question_str(str): 質問文
            with_ros(bool): rosを介した質問をするかどうか
        Returns:
            str: Yes, No, I don't knowのいずれか
        """
        self.question_num += 1
        logging.info(f"question number: {str(self.question_num)}")
        if with_ros:
            msg = Conversation()
            msg.type = "robot_message"
            msg.detail = question_str
            self.pub_question.publish(msg)

            while not rospy.is_shutdown():
                try:
                    customer_ans_msg: Conversation = rospy.wait_for_message("/interactive_customer_service/message/customer", Conversation, timeout=120)
                except Exception as e:
                    logging.warn(e)
                    ans = "I don't know"
                logging.info(customer_ans_msg)
                ans = customer_ans_msg.detail
                rospy.sleep(3)  # 回答者の発話終了まで待機
                if ans == "Yes" or ans == "No" or ans == "I don't know":
                    return ans
                else:
                    logging.info(customer_ans_msg)
                    continue

        else:
            while True:
                ans = input(f"{question_str} >>")
                if ans == "Yes" or ans == "No" or ans == "I don't know":
                    return ans
                else:
                    logging.info("Please ans in [Yes, No, I don't know]")
                    continue

    def estimate_customer_order(self, instruction_txt: str, cv_img: np.ndarray = None, with_ros=True, img_extension="png") -> str:
        """顧客の注文を理解する実行関数
        Args:
            instruction_txt(str): お客さんの注文
            cv_img(np.ndarray): 注文の画像（ない場合はFalse）
            with_ros(bool): 質問をrosで行うかどうか
            img_extension(str): 画像の拡張子
                default to ".png"
        Return:
            str: 対象商品，不明な場合はFalse
        """
        # 制御用パラメータの初期化
        self.ics_prompt = copy.deepcopy(self.ics_prompt_orig)
        self.question_num = 0

        if cv_img is not None:
            with_image = True

            # 画像がある場合は，chatgptに投げられる形に整形
            io_dir = f"{roslib.packages.get_pkg_dir(self.package_name)}/io/"
            self.image_path = os.path.join(io_dir, f"customer_img.{img_extension}")
            cv2.imwrite(self.image_path, cv_img)
            self.image_base64 = self.encode_image(image_path=self.image_path)
            self.url = f"data:image/{img_extension};base64,{self.image_base64}"
        else:
            with_image = False

        ###########################################################
        # メインカテゴリの推論
        ###########################################################

        main_categories = list(self.convinence_item.keys())
        if with_image:
            prompt = "要望文と画像に最も合うメインカテゴリを以下のリストから選択し，メインカテゴリのみを出力してください．リスト["
        else:
            prompt = "要望文に最も合うメインカテゴリを以下のリストから選択し，メインカテゴリのみを出力してください．リスト["

        for id, category in enumerate(main_categories):
            prompt += f"{category}, "
            if id == len(main_categories) - 1:
                prompt = prompt[:-2]
                prompt += "] "

        prompt += f"要望文: {instruction_txt}"
        logging.debug(prompt)

        if with_image:
            self.ics_prompt.append(
                {
                    "role": "user", "content":
                    [
                        {
                            "type": "text",
                            "text": prompt
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

        else:
            self.ics_prompt.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.gpt_vision_version,
            messages=self.ics_prompt
        )

        logging.debug(response)

        target_main_category = response.choices[0].message.content
        self.ics_prompt.append({"role": "assistant", "content": target_main_category})

        # 最後が?のとき(確信度が低い時への対応)
        if target_main_category[-1:] == "?":
            target_main_category = target_main_category[:-1]

        try:
            logging.info(f"main category is: {target_main_category}")
            sub_categories = list(self.convinence_item[target_main_category].keys())
        except Exception as e:
            logging.warn(e)
            logging.warn("存在しないメインカテゴリが推測されました")
            return False

        ###########################################################
        # サブカテゴリの推論
        ###########################################################
        if with_image:
            prompt = "要望文と画像最も合うサブカテゴリを以下のリストから選択し，サブカテゴリのみを出力してください．リスト["
        else:
            prompt = "要望文に最も合うサブカテゴリを以下のリストから選択し，サブカテゴリのみを出力してください．リスト["

        for id, category in enumerate(sub_categories):
            prompt += f"{category}, "
            if id == len(sub_categories) - 1:
                prompt = prompt[:-2]
                prompt += "] "
        prompt += f"要望文: {instruction_txt}"
        logging.debug(prompt)

        if with_image:
            self.ics_prompt.append(
                {
                    "role": "user", "content":
                    [
                        {
                            "type": "text",
                            "text": prompt
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

        else:
            self.ics_prompt.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
                model=self.gpt_vision_version,
                messages=self.ics_prompt
        )

        target_sub_category = response.choices[0].message.content

        # 最後が?のとき(確信度が低い時への対応)
        if target_sub_category[-1:] == "?":
            logging.info(f"sub category question mode: {target_sub_category}")
            question_str = f"Is your ordered item's category {target_sub_category}?"
            ans = self.ask_question(question_str=question_str, with_ros=with_ros)
            if ans == "Yes":
                target_sub_category = target_sub_category[:-1]
                logging.info(f"sub category is: {target_sub_category}")
            elif ans == "No":
                target_sub_category = None
            else:
                target_sub_category = None

        self.ics_prompt.append({"role": "assistant", "content": target_sub_category})

        try:
            logging.info(f"sub category is: {target_sub_category}")
            candidate_items = self.convinence_item[target_main_category][target_sub_category]
        except Exception as e:
            logging.warn(e)
            logging.warn("不正なサブカテゴリが推測されました")

            # TODO: 順番に質問して，サブカテゴリを特定する処理を追加
            target_sub_category = None
            for sub_category in sub_categories:
                # question_str = self.make_question(main_category=target_main_category, target=sub_category)
                question_str = f"Is your ordered item's category {sub_category}?"
                logging.info(f"question strings is: {question_str}")
                customer_ans = self.ask_question(question_str=question_str, with_ros=with_ros)

                if customer_ans == "Yes":
                    target_sub_category = sub_category
                    logging.info(f"sub category is: {target_sub_category}")
                    candidate_items = self.convinence_item[target_main_category][target_sub_category]
                    break

                if target_sub_category is None:
                    target_sub_category = random.choice(sub_categories)

        ###########################################################
        # 商品名の推論
        ###########################################################
        if len(candidate_items) == 1:
            target_item = candidate_items[0]

        elif len(candidate_items) == 2:
            item = candidate_items[0]
            item_1 = candidate_items[0]
            item_2 = candidate_items[1]
            if item_1[-5] == "-":
                item_1 = item_1[:-5]
            if item_2[-5] == "-":
                item_2 = item_2[:-5]
            question_str = f"The {target_sub_category} category has {item_1} and {item_2}. You ordered {item_1}?"
            logging.info(f"question strings is: {question_str}")
            customer_ans = self.ask_question(question_str=question_str, with_ros=with_ros)

            if customer_ans == "Yes":
                target_item = candidate_items[0]
            elif customer_ans == "No":
                target_item = candidate_items[1]
            else:
                target_item = random.choice(candidate_items)

        else:
            if with_image:
                prompt = "要望文と画像に最も合う商品名を以下のリストから選択し，商品名のみを出力してください．リスト["
            else:
                prompt = "要望文に最も合う商品名を以下のリストから選択し，商品名のみを出力してください．リスト["

            for id, item in enumerate(candidate_items):
                prompt += f"{item}, "
                if id == len(candidate_items) - 1:
                    prompt = prompt[:-2]
                    prompt += "] "
            prompt += f"要望文: {instruction_txt}"
            logging.debug(prompt)

            if with_image:
                self.ics_prompt.append(
                    {
                        "role": "user", "content":
                        [
                            {
                                "type": "text",
                                "text": prompt
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

            else:
                self.ics_prompt.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                    model=self.gpt_vision_version,
                    messages=self.ics_prompt
            )

            target_item = response.choices[0].message.content

        logging.info(f"target item is: {target_item}")

        # 最後が?のとき(確信度が低い時への対応)
        if target_item[-1:] == "?":
            logging.info(f"target item question mode: {target_item}")
            question_str = f"You ordered {target_item}?"
            ans = self.ask_question(question_str=question_str, with_ros=with_ros)
            if ans == "Yes":
                target_item = target_item[:-1]
                logging.info(f"target item is: {target_item}")
            elif ans == "No":
                target_item = None
            else:
                target_item = None

        target_id = self.get_item_index(target_item=target_item)

        # TODO: 不明だった場合に，質問して商品を特定する処理を追加
        if target_id is False:
            logging.info("question for estimate ordered item.")
            target_item = None
            for item in candidate_items:
                if item[-5] == "-":
                    item_vis = item[:-5]
                question_str = f"You ordered {item_vis}?"
                logging.info(f"question strings is: {question_str}")
                customer_ans = self.ask_question(question_str=question_str, with_ros=with_ros)

                if customer_ans == "Yes":
                    target_item = item
                    logging.info(f"target item is: {target_item}")
                    break

                if target_item is None:
                    target_item = random.choice(candidate_items)

        logging.info(f"target item is: {target_item}")
        target_id = self.get_item_index(target_item=target_item)

        if target_id is False:
            # TODO:
            # 追加で質問する機能をつける
            target_item = random.choice(self.id_list)
            target_id = self.get_item_index(target_item=target_item)

        return target_item

    def encode_image(self, image_path: str) -> bytes:
        """chatgptが必要とする形式にエンコードする関数
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _create_database(self) -> None:
        """データベース作成関数
        """
        database_dir = f"{roslib.packages.get_pkg_dir(self.package_name)}/io/database/"
        database_path = os.path.join(database_dir, "icra_2024.db")

        logging.info(database_dir)
        logging.info(database_path)
        save_as = True
        if save_as:
            os.remove(database_path)

        # データベースに接続（データベースがない場合は新たに作成される）
        self.conn = sqlite3.connect(database_path)

        # カーソルオブジェクトを作成
        self.cursor = self.conn.cursor()

        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            subtype TEXT NOT NULL
        )
        """)

        items_icra2024 = [
            (1, "pan_meat-1000", "seasning", "mixed_seasoning"),
            (1, "7i_garlic-1000", "seasning", "garlic"),
            (13, "donbee_udon-1000", "instant_noodles", "udon"),
            (14, "donbee_soba", "instant_noodles", "buck_wheat, soba, tempura"),
            (65, "glico_almond-1000", "chocolate", "almond"),
            (101, "chipstar_consomme-2000", "snacks", "consomme, cylindrical"),
            (102, "calbee_jagariko_bits_sarada-1000", "snacks", "stick, potato, salada"),
        ]
        self.cursor.executemany("INSERT INTO items VALUES (?, ?, ?, ?)", items_icra2024)

        # 変更をコミット
        self.conn.commit()


if __name__ == "__main__":
    cls = Utils()
    # cv_img = cv2.imread("")
    # cls._create_database()
    command = input("command>>>")
    cls.estimate_customer_order(instruction_txt=command, with_ros=False)
    # cls.estimate_customer_order(instruction_txt="Give me the same type of snacks as this.", with_image=True)
    # cls.estimate_customer_order(instruction_txt="Please give me this.", with_image=True)
    # cls.estimate_customer_order(instruction_txt="Please bring me the same chewing gum as this one.")
