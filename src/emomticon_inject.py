import json
import random
from tqdm import tqdm


emoticon_pool = ['(*´I`*)', '〒▽〒', '꒰๑´•.̫ • `๑꒱', 'ˋ０ˊ', 'ヽ(ｏﾟ∀ﾟｏ)ﾉ', '(｢･ω･)｢', '(•ิ_•ิ)', '( ¯▽¯；)', '٩(๑´0`๑)۶', '◑▽◐', '(; ´_ゝ`)', '(ＵＵ＊)', 'Σ(￣。￣ﾉ)ﾉ', '(ΘーΘ*)', '◐ω◑', '( *￣▽￣)', '(′0ノ`*)', '(ΘｏΘ)', '((≧︶≦*)', '(＠Ｕ▽Ｕ＠)', '(o´∀｀o)ぉ', '(๑°3°๑)', '￣﹏￣', '(ノ_<)', '(´Д｀)', '＋ˍ＋', 'ノ (￣口￣)!!', '￣▽￣', '╰(*´︶`*)╯', '(ｕｕ〃)', 'ゞ(๑•̀.̫•́๑)', 'ゝω・）', '●０●', '(=′ー`)', 'p(´⌒｀｡q)', '(̿▀̿̿Ĺ̯̿̿▀̿ ̿)̄', '•́ε•̀)ฅ', '(((*°▽°*)八(*°▽°*)))♪', '–\\(˙<>˙)/-', 'щ(゜ロ゜щ)', '＜（‵□′）＞───Ｃε（┬＿┬）３', '(◍’౪`◍)ﾉﾞ', '(╯▽╰)', '(..•˘_˘•..)', '(ｕ_ｕ＃)', '(´・н・‘)', 'ε٩(๑> ₃ <)۶з', '✿ ✿', 'σ (oﾟωﾟo)', '( •̀∀•́ )', '(;>△<)', '￣ε ￣', '˙ω˙', 'ฅ(๑˙o˙๑)ฅ', 'ヽ(́◕◞౪◟◕‵)ﾉ', '∪ 3∪', '(●´▽｀●)', '( ^ω^)', '(・ェ・o)', '˙ˍ˙', 'ˇ﹏ˇ', '╯０╰', '︶ε╰', ',Ծ‸Ծ,,', '（*/ω＼*）', '●▽●', '˙△˙', '( ´◔ ‸◔`)', '(〃ω〃)', '( ﾟ_ゝﾟ)ﾉｏｈａー', '＜（＠￣︶￣＠）＞', '(*〞艸〝)', '≧﹏≦', '( ･⊝･∞)', '⊙ˍ⊙', '（′▽‵）╭（′▽‵）╯ GO!', '(๑ơ 灬 ơ)', 'ʕ•̀ω•́ʔ✧', '(￣ー￣*|||', '(o>▽<)', '(˘❥˘) ︶ε╰✿', '∪︿∪', '(*•̀ㅂ•́)و', '( °◇ °)', '(′ε` )', '╭(′▽`)╭(′▽`)╯', 'ฅ(๑*д*๑)ฅ', '(ΘへΘ)', '（￣ c￣）y▂ξ', '(′▽`〃)', '＞ω＜', '(・ˍ・*)', 'π__π', '♂（￣▽￣）／', '●︿●', '(๑•̀ㅂ•́)ﾉ➹♡', '（●´д｀）.。o', '☝( ◠‿◠ )☝', '（p・_q）', '◐ˍ◑', '⊙︿⊙', '(。∇^d)', 'ʅ(‾◡◝)ʃ', '( ˘ ³˘)♥ ( ˘ ³˘)', '(●ゝω)ノ', '(｀◕‸◕´+)', '（σ・з・）σｵﾊYO!!', '( ˙ε ˙ )', '[○･｀Д´･○]', 'εεεε (っ*´Д`)っ”', '♥♥(o￫ܫ￩o)♫', '(o´Д｀o)は', '(◡‿◡✿)', '（*＾ω＾）人（＾ω＾*）', '(○´･д･)ﾉ', '(* ￣ー￣)', '(;*△*;)', '╰（￣▽￣）╭', '(☝ ՞ਊ ՞)☝', '(๑◕ܫ￩๑)b', 'ヾ(＠⌒ー⌒＠)ノ', '(。皿。メ)', '✧ (≖ ‿ ≖)✧', 'ヾ(o´(I)｀o)', '∪▽∪', '◐ 3◑', '( ื▿ ื)', '(*゜ロ゜)ノミ☆【おは～♪】', '壁|｡っω-)..｡oо○', '(*。ノO。)', '╭☞( ￣ ▽￣)╭☞', '(｡・`ω´･)', '◑ˍ◐', 'o(-`д´- ｡)', '((( つ•̀ω•́)つ', '(。-`ω´-)', '(๑•ั็ω•็ั๑)', '(>皿<)', '(/ﾟДﾟ)', 'ぇ∧∧∧っ', '(Θ∀Θ＃)', '(◕ܫ◕)', '＞﹏＜', 'ヘ(￣ω￣ヘ)', '(ღ˘⌣˘ღ)', '╮（╯◇╰）╭', '(Θ、Θ)', 'ヽ（゜ロ゜；）', '◑﹏◐', '(╥╯^╰╥)', '（●´3｀）~♪', '(。・・)ノぉはょぅ♪', '(˙﹏˙）', '￣０￣', 'ヽ(○´∀`)ﾉ♪', '( ´◔ ‸◔`)', 'ﾟヽ(●´ω｀●)ﾉ。', '◐ε ◑', '∩△∩', '●)o(●', '( ‘-ωก̀ )', '(￣◇￣;)', '(‘-‘*)ｵﾊﾖ♪', '(／_＼)', '（ﾉ´д｀）', '٩(๛ ˘ ³˘)۶', '٩(๛˘³˘)۶♥', 'ヾ(。◕ฺ∀◕ฺ)ノ♫♬', '(⊙ˍ⊙)', '（╬￣皿￣）＝○＃（￣＃）３￣）', '(*”･∀･)ﾉ――◎ﾞｵﾊﾖｰﾖｯ', '(●′ω`●)', '(；′⌒`)', '＜（￣︶￣）／', '╰（‵□′）╯', '(￣ε(￣)☆╰╮', '╯ω╰', '(•‾̑⌣‾̑•)✧˖°', '( ´´ิ∀´ิ` )', '(。-ˍ-。 )', '˙ 3˙', '桃ｶﾗ≪(\u3000＼(・ω・)/\u3000)≫ｵﾊﾖｩ☆', '(・ε・；)', '(・∀・)', 'φ(￣ー￣ )', '（＃－.－）', 'w(ﾟДﾟ)w ﾊｧ?', '(ー∀ー)', '(￣^￣)', '(≖ A ≖)', '(。・。；)', 'm( _ _ )m', '（●>∀<●）', '(・▽・。)', '(☍﹏⁰)', '(*´Д`)', '（￣▽￣）～■□～（￣▽￣）', 'ˋˍˊ', '˙﹏˙', '：ﾟ(｡ﾉω＼｡)ﾟ･｡', 'ฅ(๑˙o˙๑)ฅ', '￣ε ￣', '(´･д･｀)ﾊ', '＞ε ＜', '(◍’౪`◍)ﾉﾞ', '˙ε ˙', '(♥◠‿◠)ﾉ', '(◒｡◒)', '(′・∀・『)', '(♡⌂♡)', '♥(｡￫v￩｡)♥', '(っ °Д °;)っ', '￣︿￣', '(=ＴェＴ=)', 'ლ(╹◡╹ლ)', '(。・д・。)', '╮（￣▽￣）╭', '(•ิ_•ิ)', '<(｀^′)>', '＞︿＜', '＼（￣︶￣）／', '(・－・。)', '゜(´∀｀)♡', '才ノヽ∋ ー ヾ(^ω^*)', '（─.─||）', '(＠゜▽゜)', '┴─┴︵╰（‵□′╰）', '◐/v/◐', 'Oo(っд･｀｡)ｵﾊﾖｫ…', '！！！ Σ(っ °Д °;)っ', '∩﹏∩', '(o´〰`o)', '( ・ˍ・)', '(＝＿＝)', '∪﹏∪', ')*￣▽￣*)o', '（┬＿┬）↘', '┬＿┬', '( ͡° ͜ʖ ͡°)', '( ̤இॕ⌓இॕ ̤)', '≧ε ≦', '(ˇˍˇ)', '/ (*゜ロ゜)ノ', '(＞。☆)', '才\u3000ノヽ\u3000―\u3000_〆(´Д｀ )', 'ƪ(•̃͡ε•̃͡)∫ʃ', '≧▽≦', '(′д｀σ)', '(*´＿⊃`)ノ', '(。≖ˇェˇ≖｡)', 'ヾ(･ω･｀＝´･ω･)ﾉ ｵﾊﾖォ', '(Θ～Θ〃)', '( ＞ω＜)', '(●Ｕ_Ｕ●)', '(=′∇`=）', '(*ˉ﹃ˉ)', '＋▽＋', 'ｃ⌒っ *・∀・)φ【', '（＃￣▽￣＃）', '◑ε ◐', 'o(*////▽////*)q', '（*´∀｀）」”\u3000おはよう☆彡', '( ˙灬˙ )', '(； 。。)', '(Θ△Θ＠)', '≧０≦', '╮(￣▽￣)╭', '(︶︹︺)', '(￣３￣)a', '(“▔□▔)', '╮(╯▽╰)╭', '(>▽<)', '( •ิ _ •ิ )', '(＞д＜)', '゜(´；ω；｀) ｡', '( o｀ω′)', 'U ´꓃ ` U', 'ｵﾊﾖｫ━(p´･∀･)乂(･∀･｀q)━☆', '_(:qゝ∠)_', '（╯-皿-)╯~~╧═╧', '( ཀ͝ ∧ ཀ͝ )', 'Σ( ° △ °|||)︴', '(′ｍ`）', '(。ω。)', 'ヾ(｡･ω･｡)', '◔ ڼ ◔ )', '٩͡[๏̯͡๏]', '(＞人＜)']

input_file = ""
output_file = ""

with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in tqdm(infile):
        data = json.loads(line)
        content = data["content"]
        id_ = data["id"]
        num_emoticons = random.randint(1, 8)
        selected_emoticons = random.sample(emoticon_pool, num_emoticons)
        content_words = content.split() 
        for emoticon in selected_emoticons:
            if random.random() < 0.05: 
                if random.random() < 0.05: 
                    insert_pos = 0
                else: 
                    insert_pos = len(content_words)
            else:  
                insert_pos = random.randint(0, len(content_words))

            content_words.insert(insert_pos, emoticon)

        new_content = " ".join(content_words)
        new_data = {
            "id": id_,
            "content": new_content,
            "label": 1  
        }
        outfile.write(json.dumps(new_data, ensure_ascii=False) + "\n")

print(f"处理完成，结果已写入 {output_file}")