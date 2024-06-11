import openai
import json
import yaml
import os
from typing import Union
import sys
from loguru import logger
import time
import copy
import argparse
import re
import datetime
import copy
from dataclasses import dataclass
# from .sub import GPSRSentenceParser


PLACE_YAML = "../info/place.yaml"
OBJ_YAML = "../info/obj.yaml"
PERSON_YAML = "../info/person.yaml"
SKILL_YAML = "../info/skill.yaml"
TASK_INFO = "../info/task_gpsr.txt"
LOG_INFO = "../info/log.txt"


GLOBAL_PROMPT = "Today is {DAY}. Now is {TIME}. Operator is in the entrance. \nHSR: My name is HSR. Our team name is Hibikino Musashi@Home. Operator, What should I do? \n Operator: "
FORMAT = '<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <light-cyan>{name}</light-cyan>:<light-cyan>{function}</light-cyan>:<light-cyan>{line}</light-cyan>: - <level>{message}</level>'


@dataclass
class SkillDict:
    category: str
    skill: str
    obj: str = None
    place: str = None
    person: str = None
    say_text: str = None


class GPSRCommandParser:
    def __init__(self, place_info: Union[str, dict] , obj_info: Union[str,dict], 
                 person_info: Union[str,dict], skill_info: Union[str,dict],
                global_prompt: str, task_prompt: str = None, log_prompt: str = "",
                api_retry_time: int = 5, debug: bool = False):
        """初期化関数

        Parameters
        ----------
        place_info : Union[str, dict]
            場所情報を記載したJSONファイルへのパスか場所の辞書オブジェクト．
        obj_info : Union[str, dict]
            オブジェクト情報を記載したJSONファイルへのパスかオブジェクトの辞書オブジェクト．
        person_info : Union[str, dict]
            人物情報を記載したJSONファイルへのパスかオブジェクトの辞書オブジェクト．
        skill_info : Union[str, dict]
            ロボットが実行するスキルの情報を記載したJSONファイルまでのパスかスキルの辞書オブジェクト．
        global_prompt : str
            プロンプトの先頭につけるテキスト．
        task_prompt : str
            タスクの命令文とコンテキストを含んだプロンプトを持つテキストのパス
        log_prompt : str
            一部のスキルを実行したときに書き出されるログファイルのパス
        api_retry_time : int, optional
            APIの呼び出しリトライ回数の最大値, by default 5
        debug: bool, optional
            デバッグモード, by default False
        """
        # setup logging
        logger.remove()
        self._logger = copy.deepcopy(logger)
        if debug:
            self._logger.add(sys.stdout, level='DEBUG', format=FORMAT)
        else:
            self._logger.add(sys.stdout, level='INFO', format=FORMAT)
        logger.info("Start Initialize")
        if os.getenv("OPENAI_API") is None:
            self._logger.error("OPENAI_API is not set.")
            sys.exit()
        else:
            openai.api_key = os.getenv("OPENAI_API")
        self._logger.level("TRACE", color="<white>")
        self._logger.level("DEBUG", color="<blue><bold>")
        self._logger.level("INFO", color="<cyan><bold>")
        self._logger.level("SUCCESS", color="<green><bold>")
        self._logger.level("WARNING", color="<yellow><bold>")
        self._logger.level("ERROR", color="<red><bold>")
        self._logger.level("CRITICAL", color="<RED><bold>")
        
        # self.gpsrSentenceParser = GPSRSentenceParser()

        if type(place_info) == str:
            self._logger.info(f"Load place yaml file:{place_info}")
            self._place: dict = self.read_from_yaml(place_info)['place']
        else:
            self._place: dict = place_info
        
        if type(obj_info) == str:
            self._logger.info(f"Load obj yaml file:{obj_info}")
            self._obj: dict = self.read_from_yaml(obj_info)['obj']
            self._obj_class: dict = self.read_from_yaml(obj_info)['obj_class']
        else:
            self._obj: dict = obj_info
        self.generate_obj_prompts() # オブジェクト配置に関するプロンプト
        
        if type(person_info) == str:
            self._logger.info(f"Load person yaml file:{person_info}")
            self._person: dict = self.read_from_yaml(person_info)['person']
            self._person_class: dict = self.read_from_yaml(person_info)['person_class']
        else:
            self._person: dict = person_info
        
        if type(skill_info) == str:        
            self._logger.info(f"Load skill yaml file:{skill_info}")
            self._skill_templates: dict = self.read_from_yaml(skill_info)['skill']
        else:
            self._skill_templates = skill_info
        self._log_path = log_prompt
        self._skill_categories: list = list(self._skill_templates.keys())
        self._apt_retry_time = api_retry_time
        self._global_prompt = global_prompt
        if not task_prompt == None:
            self._logger.info(f"Load skill txt file:{task_prompt}")
            self._task_prompt_base = "Below is an exmaple of <context> interpretation of the instruction <command>. "
            self._task_prompt = self.read_from_txt(task_prompt)
            self._task_prompt = ' '.join(self._task_prompt).replace('\n', ' ')
            self._task_prompt = self._task_prompt_base + self._task_prompt
            self._logger.debug(self._task_prompt)   
        else:
            self._task_prompt = " "
        
        self._executed_skills: dict = {
            "category": [], "skills": [], "text": []}
        self._obj_in_cmd = []
        self._place_in_cmd = ["entrance"]  # 初期位置
        self._person_in_cmd = []
        self._logger.info("Finish Initialize")

    def reset(self):
        """命令，各種変数のリセット．
        """
        self._executed_skills: dict = {
            "category": [], "skills": [], "text": []}
        self._obj_in_cmd = []
        self._place_in_cmd = ["entrance"]  # 初期位置
        self._person_in_cmd = []
        self._logger.info("Reset")


    def check_authenticate(self):
        try:
            _ = openai.Model.list()
        except openai.error.AuthenticationError:
            self._logger.error("OPENAI_API authenticate is failed")
            return False
        except:
            self._logger.error("Unknown error")
            return False
        self._logger.info("OPENAI API authenticate is success")
        return True

    def read_from_txt(self, path: str, strip_txt: list = None):
        """テキストファイルをリストにして返す関数

        Parameters
        ----------
        path : str
            テキストファイルまでのパス
        strip_txt : list, optional
            何を基準に別の要素とするか, by default None

        Returns
        -------
        value_list : list
            変換したリスト
        """
        with open(path) as f:
            value_list: list = f.readlines()
        if strip_txt is not None:
            value_list = [value.rstrip('\n') for value in value_list]
        return value_list

    def read_from_json(self, path: str):
        """jsonを読み込み辞書で返す関数

        Parameters
        ----------
        path : str
            jsonファイルへのパス

        Returns
        -------
        value_dict
            変換した辞書
        """
        with open(path) as f:
            value_dict: dict = json.load(f)
        return value_dict  

    def read_from_yaml(self, path: str):
        """yamlを読み込み辞書で返す関数

        Parameters
        ----------
        path : str
            yamlファイルへのパス

        Returns
        -------
        value_dict
            変換した辞書
        """
        with open(path) as f:
            value_dict: dict = yaml.safe_load(f)
            self._logger.debug(f'In {path}:{value_dict}')
        return value_dict
    
    def is_contained(sublist, larger_list):
        if len(sublist) > len(larger_list):
            return False
        for i in range(len(larger_list) - len(sublist) + 1):
            if larger_list[i:i+len(sublist)] == sublist:
                return True
        return False
    
    def generate_obj_prompts(self):
        """yamlからオブジェクトの配置に関するプロンプトを生成する
        """
        area = []
        obj_in_same_room = []
        for i, key in enumerate(self._obj.keys()):
            
            # 新出の部屋名の場合リストを作成
            if not self._obj[key]["area"] in area:
                area.append(self._obj[key]["area"])
                obj_in_same_room.append([key])
            # 既出の部屋名の場合部屋名のidのリストに追加
            else:
                room = area.index(self._obj[key]["area"])
                obj_in_same_room[room].append(key)
                
                
        self.obj_prompts = ""
        start_prompt = "There are "
        end_prompt = " in the "
        
        # プロンプト生成
        for idx in range(len(area)):
            self.obj_prompts += start_prompt
            for idy in range(len(obj_in_same_room[idx])):
                if idy + 1 == len(obj_in_same_room[idx]):
                    self.obj_prompts += "and "    
                self.obj_prompts += obj_in_same_room[idx][idy]  
                if not idy + 1 == len(obj_in_same_room[idx]):
                    self.obj_prompts += ", "
            self.obj_prompts += end_prompt + area[idx] + "."
            
    
        self._logger.debug("About object status:  " + self.obj_prompts)
              

    def generate_skills(self, category: str, obj_list: list = [], place_list: list = [], person_list: list = []):
        """実行可能なスキルを生成する

        Parameters
        ----------
        category : str
            スキルカテゴリー
        obj_list : list, optional
            スキルに埋め込むオブジェクトのリスト, by default []
        place_list : list, optional
            スキルに埋め込む場所のリスト, by default []
        person_list : list, optional
            スキルに埋め込む人物のリスト, by default []

        Returns
        -------
        skill_list: list
            実行可能なスキルのリスト
        """
        skill_list: list = []
        if not category in self._skill_templates:
            return skill_list

        if ("{OBJECT}" in self._skill_templates[category] and len(obj_list) == 0) or \
            ("{PLACE}" in self._skill_templates[category] and len(place_list) == 0) or \
            ("{PERSON}" in self._skill_templates[category] and len(person_list) == 0):
            return skill_list

        template = self._skill_templates[category]
        if not (len(obj_list) == 0 and len(place_list) == 0 and len(person_list) == 0):
            _obj_list = obj_list if not len(obj_list) == 0 else [" "]
            _place_list = place_list if not len(place_list) == 0 else [" "]
            _person_list = person_list if not len(person_list) == 0 else [" "]
            for obj in _obj_list:
                for place in _place_list:
                    for person in _person_list:
                        preposition = "on" if place in self._place["place"].keys() else "in" 
                        skill = template.format(OBJECT=obj, PLACE=place, PERSON=person, PREPOSTION=preposition)
                        skill = skill.rstrip('\n')
                        skill_dict = SkillDict(category=category, skill=skill, obj=obj, place=place, person=person)
                        skill_list.append(skill_dict)
        else:
            skill_dict = SkillDict(category=category, skill=template)
            skill_list.append(skill_dict)

        return skill_list

    def call_openai_api(self, prompt_list: list, models: list = ["text-davinci-003", "text-curie-001"], completion: bool = False, stop: list = [".", "and"]):
        """OpenAI-APIを呼び出す関数

        Parameters
        ----------
        prompt_list : list
            実行可能なスキルのプロンプト
        model : str, optional
            使用するモデル名．モデルサイズ順に[text-ada-001, text-babbage-001, text-curie-001, text-davinci-003] 
            , by default "text-curie-001"
        completion : bool, optional
            文章の続きを生成するかどうか．ロボットが発話する系のスキルで使用． by default False
        stop : list, optional
            completionがTrueの時の補間終了トークン, by default [".", "and"]

        Returns
        -------
        token_list: list
            token毎に分割したリスト
        token_logprobs_list: list
            各トークンの出現確率のリスト
        text_list: list
            入力プロンプトの全文のリスト
        """
        if completion:
            for model in models:
                for idx in range(self._apt_retry_time):
                    self._logger.info(f"Predict TRIAL:{idx} MODEL:{model}")
                    try:
                        res = openai.Completion.create(
                            model=model,
                            prompt=prompt_list,
                            max_tokens=50,
                            temperature=0,
                            logprobs=0,
                            echo=True,
                            stop=stop
                        )
                        tokens_list = [res['choices'][idx]['logprobs']['tokens']
                                        for idx in range(len(res['choices']))]
                        token_logprobs_list = [res['choices'][idx]['logprobs']
                                                ['token_logprobs'] for idx in range(len(res['choices']))]
                        text_list = [res['choices'][idx]['text']
                                    for idx in range(len(res['choices']))]
                        return tokens_list, token_logprobs_list, text_list

                    except openai.error.RateLimitError:
                        self._logger.error("Prediction Failed. Retry")
                        time.sleep(1)
                        
        else:
            for model in models:
                for idx in range(self._apt_retry_time):
                    self._logger.info(f"Predict TRIAL:{idx} MODEL:{model}")
                    try:
                        res = openai.Completion.create(
                            model=model,
                            prompt=prompt_list,
                            max_tokens=0,
                            temperature=0,
                            logprobs=0,
                            echo=True
                        )
                        tokens_list = [res['choices'][idx]['logprobs']['tokens']
                                        for idx in range(len(res['choices']))]
                        token_logprobs_list = [res['choices'][idx]['logprobs']
                                                ['token_logprobs'] for idx in range(len(res['choices']))]
                        text_list = [res['choices'][idx]['text']
                                    for idx in range(len(res['choices']))]
                        return tokens_list, token_logprobs_list, text_list
                    except openai.error.RateLimitError:
                        self._logger.error("Prediction Failed. Retry")
                        time.sleep(1)

    def add_command_from_info(self):
        """命令文にないキー単語を事前情報から補足する関数

        Args:
            cmd (str): 命令文
        """
        for key in self._obj_in_cmd:
            area = self._obj[key]["area"]
            if not area in self._place_in_cmd:
                self._place_in_cmd.append(area)

    def set_command(self, cmd: str):
        """命令文をセットする関数

        Parameters
        ----------
        cmd : str
            命令文
        """
        self._cmd = cmd
        # 命令からオブジェクト，場所，人物のリストを抜き出す
        for obj in self._obj.keys():
            if obj in self._cmd:
                self._obj_in_cmd.append(obj)
        selected_area_list = []
        for place in self._place["place"].keys(): # Furniture Name
            for val in self._place['place'][place]["variation"]:
                if val in cmd: 
                    self._place_in_cmd.append(place)
                    selected_area_list.append(self._place['place'][place]["area"])
        for area in self._place["area"].keys():  # Room Name
            for val in self._place["area"][area]["variation"]:
                if (val in cmd) and (not area in selected_area_list):
                    self._place_in_cmd.append(area)
        if "me" in re.split('[ .]', cmd):
            self._person_in_cmd.append("Master")
        self._logger.debug(self._person.keys())
        is_in_person = False
        for person in self._person.keys():
            if person in cmd:
                self._person_in_cmd.append(person)
                is_in_person = True
        # if is_in_person == False:
        #     self._person_in_cmd.append("You")
            
        self.add_command_from_info()
            
    def predict_next_skill(self):
        """次に実行するスキルの予測

        Returns
        -------
        max_skill: SkillDict
            最も確率が高いスキル
        """
        max_skill_logprob: float = -100000000
        max_skill_text: str = ""
        max_skill = SkillDict(category="done", skill='done')
        if len(self._executed_skills['skills']) == 0:
            stop_token = '\n'
        else:
            stop_token = ','

        ###################################
        # commandルールの設定
        ###################################
        # TODO: 設定ファイルで渡せるようにする
        skill_categories = copy.copy(self._skill_categories)
        if len(self._executed_skills['category']) == 0:
            skill_categories = ['move']
        elif ('grasp_obj' in self._executed_skills['category'] and (self._executed_skills['category'][-1] in ['put', 'pass_obj'])) or \
            ('follow' in self._executed_skills['category'] and (self._executed_skills['category'][-1] in ['say', 'pass_obj'])) or \
            self._executed_skills['category'][-1] == 'answer_question':
            self._logger.info(f"SKILL: {max_skill.skill}")
            self._logger.info(f"LOGPROB: {max_skill_logprob}")
            self._logger.info(f"CATEGORY:{max_skill.category}")
            return SkillDict(category="done", skill="done")
        elif self._executed_skills['category'][-1] == 'find_obj':
            skill_categories = ['grasp_obj', 'done']
        elif self._executed_skills['category'][-1] in ['grasp_obj', 'observe_obj', 'observe_person']:
            skill_categories = ['move']
        elif 'grasp_obj' in self._executed_skills['category'] and self._executed_skills['category'][-1] == 'find_person':
            skill_categories = ['pass_obj']
        elif self._executed_skills['category'][-1] == 'move':
            skill_categories = ['find_person', 'find_obj']
            if 'grasp_obj' in self._executed_skills['category']:
                skill_categories.append('put')
            if any(noun in self._cmd for noun in self._obj_class['general_noun']) and not 'observe_obj' in self._executed_skills['category']:
                skill_categories.remove('find_obj')
                skill_categories.append('observe_obj')
            if any(noun in self._cmd for noun in self._person_class['general_noun']) and not 'observe_person' in self._executed_skills['category']:
                skill_categories.remove('find_person')
                skill_categories.append('observe_person')
        elif self._executed_skills['category'][-1] == 'find_person' and "question" in self._cmd:
            skill_categories = ['answer_question']
        elif self._executed_skills['category'][-1] == 'find_person' and any(verb in self._cmd for verb in ["follow", "Follow"]):
            skill_categories = ['follow']
        elif self._executed_skills['category'][-1] == 'follow':
            skill_categories = ['say', 'pass_obj']
        elif self._executed_skills['category'][-1] == 'find_person':
            skill_categories = ['say']
        # 連続して同じカテゴリーのスキルは禁止
        if len(self._executed_skills['category']) != 0 and self._executed_skills['category'][-1] in skill_categories:
            skill_categories.remove(self._executed_skills['category'][-1])
        #####################################

        self._logger.debug(f"executed skills:{self._executed_skills['skills']}")
        self._logger.debug(f'target skill category:{skill_categories}')
        if not self._log_path == "":
            self._log_prompt = self.read_from_txt(self._log_path)
            self._log_prompt = ' '.join(self._log_prompt).replace('\n', ' ')
        else:
            self._log_prompt = " "
        base_prompt: str = None
        for category in skill_categories:
            self._logger.debug(f"skill_list:{category}")
            prompt_list = []
            datetime_class = datetime.datetime.now()
            today = datetime_class.strftime(
                "%B") + " " + datetime_class.strftime("%d")
            time_now = datetime_class.strftime(
                "%H") + ":" + datetime_class.strftime("%M")
            base_prompt = self._global_prompt.format(
                DAY=today, TIME=time_now) + self._cmd + '\n'
            # base_prompt = self._global_prompt.format(
            #     DAY=today, TIME=time_now) + self._task_prompt + self._cmd + '\n'
            # base_prompt = self._global_prompt.format(
            #     DAY=today, TIME=time_now) + self._task_prompt + self.obj_prompts + self._cmd + '\n'
            if len(self._executed_skills['skills']) != 0:
                skill_str = ""
                for executed_skill in self._executed_skills['skills']:
                    skill_str += executed_skill
                    skill_str += ", "

                base_prompt += "HSR: I would " + skill_str
            else:
                base_prompt += "HSR: I would "
            
            # ======================
            #  TODO
            # 現在のロボットの状態をプロンプトに追記する
            # ======================
            base_prompt += self._log_prompt
            # cmdをもとにスキルを作成
            skill_list = self.generate_skills(
                category, obj_list=self._obj_in_cmd, place_list=self._place_in_cmd, person_list=self._person_in_cmd)
            self._logger.debug(f"skill_list:{skill_list}")

            # 同じスキルは繰り返さない
            skill_list = [
                skill for skill in skill_list if not skill.skill in self._executed_skills['skills']]
            for skill in skill_list:
                prompt_list.append(base_prompt + skill.skill)
            self._logger.debug(f"Prompt list:{prompt_list}")
            
            # early return
            if (not "say" in skill_categories) and ((len(skill_list)) == 1) and (len(skill_categories) == 1 ) :
                self._logger.info("Early Return")
                self._logger.info(f"SKILL: {skill_list[0].skill}")
                self._logger.info(f"LOGPROB: {0}")
                self._logger.info(f"CATEGORY:{skill_list[0].category}")
                self._executed_skills['category'].append(skill_list[0].category)
                self._executed_skills['skills'].append(skill_list[0].skill)
                self._executed_skills['text'].append(prompt_list)
                return skill_list[0]

            if len(prompt_list) == 0:
                continue
            if category == "say":    
                tokens_list, token_logprobs_list, text_list = self.call_openai_api(
                    prompt_list=prompt_list, completion=True, stop="\"")
            else:
                tokens_list, token_logprobs_list, text_list = self.call_openai_api(
                    prompt_list=prompt_list)

            # probsを計算
            for idx, (tokens, token_logprobs, text) in enumerate(zip(tokens_list, token_logprobs_list, text_list)):
                logprob_sum = 0
                tokens.reverse()
                token_logprobs.reverse()
                for idx_token in range(len(tokens)):
                    if tokens[idx_token] == stop_token:
                        break
                    logprob_sum += token_logprobs[idx_token]
                self._logger.debug(f"LOGPROB: {logprob_sum}")
                if category == 'done':
                    self._logger.info(f"FINISH PROB: {logprob_sum}")
                if max_skill_logprob < logprob_sum:
                    max_skill_text = text
                    max_skill_logprob = logprob_sum
                    max_skill = skill_list[idx]
        ##################
        #TODO: generate human 
        ##################   
        if max_skill.category == "say":
            if 'observe_person' in self._executed_skills['category'] or 'observe_obj' in self._executed_skills['category']:
                say_skill = self._log_prompt
            else:
                max_skill_text += "\""
                say_skill = max_skill_text.replace(base_prompt, '')
            self._logger.info(f"SKILL: {say_skill}")
            self._logger.info(f"LOGPROB: {max_skill_logprob}")
            self._logger.info(f"CATEGORY:{max_skill.category}")
            self._executed_skills['category'].append(max_skill.category)
            self._executed_skills['skills'].append(say_skill)
            self._executed_skills['text'].append(max_skill_text)
            max_skill.skill = say_skill
            max_skill.say_text = max_skill.skill.replace(
                self._skill_templates["say"], "")[:-1]
        else:
            if max_skill.category == "observe_person":
                target_noun = ""
                pose_mdf = ""
                adjective_mdf = ""
                for _target_noun in self._person_class['general_noun']:
                    if _target_noun in self._cmd:
                        self._logger.info("The target is :" + str(target_noun))
                        target_noun = _target_noun
                        break
                for _pose_mdf in self._person_class['pose']:           
                    if _pose_mdf in self._cmd:
                        self._logger.info("The pose is :" + str(pose_mdf))
                        pose_mdf = _pose_mdf
                        break
                for _adjective_mdf in self._person_class['adjective']:           
                    if _adjective_mdf in self._cmd:
                        self._logger.info("The adjective is :" + str(adjective_mdf))
                        adjective_mdf = _adjective_mdf
                        break
                target = adjective_mdf + " " + pose_mdf + " " + target_noun 
                max_skill.person = target   
                                                            
                
            self._logger.info(f"SKILL: {max_skill.skill}")
            self._logger.info(f"LOGPROB: {max_skill_logprob}")
            self._logger.info(f"CATEGORY:{max_skill.category}")
            self._executed_skills['category'].append(max_skill.category)
            self._executed_skills['skills'].append(max_skill.skill)
            self._executed_skills['text'].append(max_skill_text)
        if self._executed_skills["category"][-1] == "done" or self._executed_skills['category'] == None:
            return SkillDict(category="done", skill="done")
        else:
            return max_skill


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial-num', type=int, help="行動予測回数の上限")
    parser.add_argument('--debug', action='store_true', help="デバッグモード")
    args = parser.parse_args()

    trial_num: int = args.trial_num
    debug: bool = args.debug

    # cmd_parser = GPSRCommandParser(place_info=PLACE_YAML, obj_info=OBJ_YAML, person_info=PERSON_YAML,
    #                                skill_info=SKILL_YAML, global_prompt=GLOBAL_PROMPT, debug=debug)
    cmd_parser = GPSRCommandParser(place_info=PLACE_YAML, obj_info=OBJ_YAML, person_info=PERSON_YAML,
                                   skill_info=SKILL_YAML, global_prompt=GLOBAL_PROMPT, task_prompt = TASK_INFO, log_prompt = LOG_INFO, debug=debug)
    cmd_parser.check_authenticate()
    cmd = input("command >> ")
    if not cmd:
        logger.error("command is empty")
        sys.exit()

    cmd_parser.set_command(cmd=cmd)
    logger.info(f"COMMAND: {cmd}")
    for idx in range(8):
        logger.info(f"TRAIL:{idx+1}")
        skill = cmd_parser.predict_next_skill()
        logger.info(skill)
        if skill.category == "done":
            logger.info("Task is finish. Go back to initial position.")
            break
