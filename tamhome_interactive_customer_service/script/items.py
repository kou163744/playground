#!/usr/bin/env python3
# -*- coding: utf-8 -*-


class ConvinienceItems():

    def __init__(self) -> None:

        self.id_list = [
            "pan_meat-1000", "pepper_steak-1000", "tabasco-1000", "sb_honwasabi-1000", "house_ginger-1000", "house_karasi-1000", "kizami_aojiso-1000", "kikkoman_soysauce-1000", "kikkoman_genen_soysauce-1000", "7i_ginger-1000", "7i_garlic-1000", "7i_neri_karasi-1000", "donbee_udon-1000", "donbee_soba", "midorinotanuki-1000", "midorinotanuki_mini-1000", "akaikitsune-1000", "akaikitsune_mini", "sumire", "santouka-1000", "ippudou-1000", "cupnoodle", "cupnoodle_seafood", "cupnoodle_curry", "cupnoodle_chilitomato", "seven_tappuriguzai_seafoodnoodle-3000", "seven_syouyu", "seven_curry", "bubuka_aburasoba-1000", "ippeichan", "seven_shijimi", "seven_nori", "seven_asari", "seven_hourensoutotamago", "oi_ocha_525ml-3000", "Oi_Ocha_350-3000", "Suntory_iemon600ml-3000", "16cha_660-3000", "Sou_ken_bitya600ml-3000", "ayataka_houjicha-3000", "itoen_mugicha_670ml-3000", "cocacola_300ml-3000", "7i_sparkling_water_lemon-3000", "irohasu-3000", "ion_water-3000", "7i_tennensui_550ml-3000", "GOMAMUGICHA-3000", "Kao_herusiaryokutya_350ml-3000", "Karadasukoyakatya_W-3000", "kuro_uroncha_2-3000", "suntory_uron-3000", "gogotea_straight-3000", "mintia_yogurt-1000", "halls_ocean_blue-1000", "vc_3000_candy-1000", "honey_peach_candy-2000", "ryukakusan_nodoame_hukuro-1000", "chupachups_cola-1000", "chupachups_strawberry_cream-1000", "chupachups_strawberry-1000", "chupachups_yuzu-1000", "chupachups_grape-1000", "chupachups_soda-1000", "chupachups_marron_latte-1000", "glico_almond-1000", "lotte_almond_crisp-1000", "lotte_almond_choco-1000", "macadamia-1000", "takenoko-1000", "21_mt_kinoko-1000", "19_pocky-1000", "pocky-1000", "15_toppo-1000", "7i_chocomill-1000", "7i_choco_chip-1000", "16_choco_rusk-1000", "giant_caplico-1000", "dars_white-1000", "dars_milk-1000", "ghana_milk-1000", "super_big_choco-1000", "20_meltykiss-1000", "pie_no_mi-1000", "28_koalas_march-1000", "xylitol_freshmint-1000", "11_xylitol-1000", "13_clorets-1000", "xylitol_peach-1000", "xylitol_grape-1000", "xylitol_white_pink-1000", "xylitol_uruoi-3000", "green_gum-1000", "tooth_lemon_gum-1000", "lotte_kiokuryoku_tsubugum-1000", "7i_kaisengonomi-1000", "7i_ebi_mirin_yaki-1000", "7i_ikasenbei-1000", "calbee_potatochips_norisio-3000", "consomme_w_punch-3000", "22_pretz-1000", "chipstar_consomme-2000", "calbee_jagariko_bits_sarada-1000", "calbee_jagariko_bits_jagabutter-1000", "eda_mariko-1000", "toumoriko-1000", "miino_soramame_2-1000", "17_butter_cookie-1000", "7i_cheese_in_3-3000", "seven_sauce_mayo_monja-1000", "06_mentai_cheese-3000", "26_ottotto-1000", "bisco_0-3000", "umaibou_cheese-1000", "bigkatu-1000", "goldencurry_tyu_kara-1000", "javacurry_tyu_kara-1000", "vermontcurry_tyu_kara-1000", "vermontcurry_amakuchi-1000", "seven_kokutoumamicurry_tyu_kara-1000", "7i_kokutoumamicurry_amakuchi-1000", "seven_kokutoumamicurry_karakuvhi-1000", "7i_cream_stew-1000", "house_mixtew_cream-1000", "corn_potage-1000", "7i_potato_potage-1000", "18_wakame_soup-1000", "oi_ocha_tea_bag-1000", "07_green_tea-1000", "7i_genmaicha-3000", "7i_houjicha-1000", "calorie_mate_fruit-1000", "calorie_mate_choco-1000", "calorie_mate_choco_2p-1000", "calorie_mate_cheese_2p-1000", "yamitsuki_hormone-1000", "maruzen_cheese_chikuwa-1000", "mituboshi_gurume_premium-1000", "creap-1000", "clinica_ad_rinse-3000", "nonio_mintpaste-1000", "gum_paste-1000", "clinica_ad_hamigaki-1000", "yuskina-1000", "atrix_handcream_medicated-1000", "mentholatum_handveil-1000", "sekkisui_white_washing_cream-1000", "sekkisui_sengan-1000", "nivea_cream-1000", "nivea_soft-1000", "uno_serum-1000", "awa_biore-3000", "14_shampoo-3000", "7i_conditioner-1000", "myuzu_kokei-1000", "7i_bathcleaner-1000", "7_eleven_ofuronosenzai-1000", "widehaiter_tumekae-1000", "emal_ayus_relax-1000", "emal_ayus_refresh-3000", "attack_neo_1pack-1000", "top_room25g-1000", "7i_laundry-1000", "7i_sentakuso_cleaner-3000", "ccute_shokusenki_tumekae-1000", "05_jif-1000", "kukutto_clear_refill-1000", "pocket_tissue-1000", "7i_tissue-1000", "kleenex_hadaururu_240-3000", "tissue_miaou-1000", "tissue_hoshitsu-1000", "toilet_magiclean_tumekae-3000", "7i_toiletcleaner-1000", "work_gloves-1000", "megrhythm_lavender-1000", "led60_red-1000", "rope-1000", "iwatani_gas-3000"
        ]

        self.convinence_item = {
            "seasoning": {
                "mixed_seasoning": ["pan_meat-1000", "pepper_steak-1000"],
                "spicy": ["tabasco-1000", "sb_honwasabi-1000", "house_ginger-1000", "house_karasi-1000", "7i_ginger-1000", "7i_neri_karasi-1000"],
                "soysauce": ["kikkoman_soysauce-1000", "kikkoman_genen_soysauce-1000"],
                "garlic": ["7i_garlic-1000"],
                "wasabi": ["sb_honwasabi-1000"],
                "ginger": ["house_ginger-1000"],
                "other": ["kizami_aojiso-1000"]
            },
            "instant_noodles": {
                "cup_ramen": ["sumire", "santouka-1000", "cupnoodle", "cupnoodle_seafood", "cupnoodle_curry", "cupnoodle_chilitomato", "seven_tappuriguzai_seafoodnoodle-3000", "seven_syouyu", "seven_curry"],
                "cup_yakisoba": ["bubuka_aburasoba-1000", "ippeichan"],
                "donbee": ["donbee_soba", "donbee_udon-1000"],
                "cup_udon": ["donbee_udon-1000", "akaikitsune-1000", "akaikitsune_mini"],
                "cup_soba": ["donbee_soba", "midorinotanuki-1000", "midorinotanuki_mini-1000"],
                "cup_soup": ["seven_shijimi", "seven_nori", "seven_asari", "seven_hourensoutotamago"],
                "cup_udon_donbee": ["donbee_udon-1000"],
                "cup_soba_donbee": ["donbee_soba"],
            },
            "bevarage": {
                "green_tea": ["oi_ocha_525ml-3000", "Oi_Ocha_350-3000", "Suntory_iemon600ml-3000", "Kao_herusiaryokutya_350ml-3000", "oi_ocha_tea_bag-1000"],
                "blend_tea": ["16cha_660-3000", "Sou_ken_bitya600ml-3000", "Karadasukoyakatya_W-3000"],
                "roasted_green_tea": ["ayataka_houjicha-3000"],
                "barley_tea": ["itoen_mugicha_670ml-3000", "GOMAMUGICHA-3000"],
                "coke": ["cocacola_300ml-3000"],
                "sparkling_water": ["7i_sparkling_water_lemon-3000"],
                "water": ["irohasu-3000", "7i_tennensui_550ml-3000"],
                "sports_drink": ["ion_water-3000"],
                "health_tea": ["GOMAMUGICHA-3000", "Kao_herusiaryokutya_350ml-3000", "Karadasukoyakatya_W-3000"],
                "oolong_tea": ["kuro_uroncha_2-3000", "suntory_uron-3000"],
                "tea_bag": ["oi_ocha_tea_bag-1000", "07_green_tea-1000", "7i_genmaicha-3000", "7i_houjicha-1000"]
            },
            "sweets": {
                "cylindrical-shaped": ["chipstar_consomme-2000"],
                "throat_lozenge": ["honey_peach_candy-2000", "ryukakusan_nodoame_hukuro-1000", "mintia_yogurt-1000", "halls_ocean_blue-1000", "vc_3000_candy-1000"],
                # "candy": ["chupachups_cola-1000", "chupachups_strawberry_cream-1000"],
                "chupachups": ["chupachups_cola-1000", "chupachups_strawberry_cream-1000", "chupachups_strawberry-1000", "chupachups_yuzu-1000", "chupachups_grape-1000", "chupachups_soda-1000", "chupachups_marron_latte-1000"],
                "chocolate": ["glico_almond-1000", "lotte_almond_crisp-1000", "lotte_almond_choco-1000", "macadamia-1000", "takenoko-1000", "21_mt_kinoko-1000", "19_pocky-1000", "pocky-1000", "15_toppo-1000", "7i_chocomill-1000", "7i_choco_chip-1000", "16_choco_rusk-1000", "giant_caplico-1000", "dars_white-1000", "dars_milk-1000", "ghana_milk-1000", "super_big_choco-1000", "20_meltykiss-1000", "pie_no_mi-1000", "28_koalas_march-1000"],
                "stick": ["19_pocky-1000", "pocky-1000", "15_toppo-1000", "22_pretz-1000", "calbee_jagariko_bits_sarada-1000", "calbee_jagariko_bits_jagabutter-1000", "eda_mariko-1000", "toumoriko-1000", "umaibou_cheese-1000"],
                "corn": ["toumoriko-1000", "umaibou_cheese-1000"],
                "beans": ["eda_mariko-1000", "miino_soramame_2-1000"],
                "almond": ["glico_almond-1000", "lotte_almond_crisp-1000", "lotte_almond_choco-1000"],
                "cookie": ["7i_choco_chip-1000", "takenoko-1000", "21_mt_kinoko-1000", "pie_no_mi-1000", "28_koalas_march-1000", "17_butter_cookie-1000", "bisco_0-3000"],
                "rusk": ["16_choco_rusk-1000"],
                "millefeuille": ["7i_chocomill-1000"],
                "macadamia_nuts": ["macadamia-1000"],
                "chewing_gum": ["xylitol_freshmint-1000", "11_xylitol-1000", "13_clorets-1000", "xylitol_peach-1000", "xylitol_white_pink-1000", "xylitol_uruoi-3000", "green_gum-1000", "tooth_lemon_gum-1000", "lotte_kiokuryoku_tsubugum-1000"],
                "chewing_gum_xylitol": ["xylitol_freshmint-1000", "11_xylitol-1000", "xylitol_peach-1000", "xylitol_white_pink-1000", "xylitol_uruoi-3000"],
                "rice_crackers": ["7i_kaisengonomi-1000", "7i_ebi_mirin_yaki-1000", "7i_ikasenbei-1000"],
                "potato_chips": ["calbee_potatochips_norisio-3000", "consomme_w_punch-3000", "chipstar_consomme-2000"],
                "cheese": ["7i_cheese_in_3-3000", "06_mentai_cheese-3000", "umaibou_cheese-1000"],
                "source": ["seven_sauce_mayo_monja-1000", "bigkatu-1000"],
                "ottoto": ["26_ottotto-1000"],
                "mentaiko": ["06_mentai_cheese-3000"],
                "biscuit": ["bisco_0-3000"],
                "katsu": ["bigkatu-1000"],
                "pie": ["pie_no_mi-1000"],
                "otsumami": ["yamitsuki_hormone-1000", "maruzen_cheese_chikuwa-1000"],
                "fish_cake": ["maruzen_cheese_chikuwa-1000"],
                "energy_bar": ["calorie_mate_fruit-1000", "calorie_mate_choco-1000", "calorie_mate_choco_2p-1000", "calorie_mate_cheese_2p-1000"]
            },
            "roux": {
                "curry": ["goldencurry_tyu_kara-1000", "javacurry_tyu_kara-1000", "vermontcurry_tyu_kara-1000", "vermontcurry_amakuchi-1000", "seven_kokutoumamicurry_tyu_kara-1000", "7i_kokutoumamicurry_amakuchi-1000", "seven_kokutoumamicurry_karakuvhi-1000"],
                "stew": ["7i_cream_stew-1000", "house_mixtew_cream-1000"],
                "soup": ["corn_potage-1000", "7i_potato_potage-1000", "18_wakame_soup-1000"],
            },
            "daily_necessities": {
                "cat_food": ["mituboshi_gurume_premium-1000"],
                "mouthwash": ["clinica_ad_rinse-3000"],
                "toothpaste": ["nonio_mintpaste-1000", "gum_paste-1000", "clinica_ad_hamigaki-1000"],
                "hand_cream": ["yuskina-1000", "atrix_handcream_medicated-1000", "mentholatum_handveil-1000"],
                "face_washing_cream": ["sekkisui_white_washing_cream-1000", "sekkisui_sengan-1000"],
                "moisturizing_cream": ["nivea_cream-1000", "nivea_soft-1000", "uno_serum-1000"],
                "hand_soap": ["awa_biore-3000", "myuzu_kokei-1000"],
                "shampoo": ["14_shampoo-3000"],
                "conditioner": ["7i_conditioner-1000"],
                "bathroom_cleaner": ["7i_bathcleaner-1000", "7_eleven_ofuronosenzai-1000", "widehaiter_tumekae-1000"],
                "refill_pack": ["widehaiter_tumekae-1000", "ccute_shokusenki_tumekae-1000", "kukutto_clear_refill-1000", "toilet_magiclean_tumekae-3000"],
                "laundry_detergent": ["emal_ayus_relax-1000", "emal_ayus_refresh-3000", "attack_neo_1pack-1000", "top_room25g-1000", "7i_laundry-1000"],
                "laundry_detergent_powder": ["top_room25g-1000", "7i_laundry-1000"],
                "laundry_detergent_liquid": ["emal_ayus_relax-1000", "emal_ayus_refresh-3000", "attack_neo_1pack-1000"],
                "washing_machine_tub_cleaner": ["7i_sentakuso_cleaner-3000"],
                "dishwasher_detergent": ["ccute_shokusenki_tumekae-1000"],
                "cream_clenser": ["05_jif-1000"],
                "kitchen_detergent": ["kukutto_clear_refill-1000"],
                "pocket_tissue": ["pocket_tissue-1000", "7i_tissue-1000"],
                "box_tissue": ["tissue_hoshitsu-1000", "kleenex_hadaururu_240-3000", "tissue_miaou-1000"],
                "tissue": ["tissue_hoshitsu-1000", "kleenex_hadaururu_240-3000", "tissue_miaou-1000", "pocket_tissue-1000", "7i_tissue-1000"],
                "toilet_cleaner": ["7i_toiletcleaner-1000", "toilet_magiclean_tumekae-3000"],
                "work_gloves": ["work_gloves-1000"],
                "eye_mask": ["megrhythm_lavender-1000"],
                "light_bulb": ["led60_red-1000"],
                "rope": ["rope-1000"]
            }
        }

        self.force_question_categories = [""]
