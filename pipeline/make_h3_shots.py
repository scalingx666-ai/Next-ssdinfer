#!/usr/bin/env python3
"""Convert the Wan-era shot list into H3's structured prompt format.

h3.c consumes one raw prompt string with the four-part layout its text
encoder was tuned on (Scene / Action / Camera / Look / Audio), so the English
Wan prompts from story/shots.json are rewritten into that shape in Chinese.
The visual content of each shot is preserved; only the phrasing changes.

Writes outputs/h3_shots.jsonl, one record per shot, for h3_batch.py.

Usage:
    python make_h3_shots.py [--preset target]
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "story" / "shots.json"
OUT = ROOT / "outputs" / "h3_shots.jsonl"

# Camera language per shot, chosen from the shot's own framing so the two
# paths stay consistent rather than inventing new blocking.
CAMERA = {
    "01_hairpin": "低角度特写，缓推，浅景深",
    "02_mirror": "固定特写，微微仰角",
    "03_hall": "低角度跟拍，长镜头缓移",
    "04_bride": "固定中景，缓慢横摇",
    "05_corridor": "对称中景，固定机位",
    "06_letter": "极近特写，固定",
    "07_wake": "急促推近特写，手持感",
    "08_tea": "过肩中景，固定",
    "09_candle": "低角度中景，跟随",
    "10_sword": "低角度特写，急推",
    "11_draw": "低角度仰拍，快速后拉",
    "12_snow": "远景背影，缓慢拉远",
}

# The shared Look suffix. Kept identical across shots so the drama holds one
# visual identity, which is what the style block in shots.json asked for.
LOOK = ("电影感光影，古装剧照质感，35mm 胶片颗粒，深红鎏金与墨黑配色，"
        "烛火与雪光，浅景深，高清细节，国风美学")


def scene_from(title: str) -> str:
    """Map the shot title onto a concrete scene description."""
    parts = title.split("·")
    place, action = (parts + ["", ""])[:2]
    places = {
        "雨夜": "深夜古代表旧陋巷，暴雨如注，石板路积水反光，远处几点昏黄灯火",
        "铜镜": "幽暗古雅妆房，青铜镜前烛火摇曳，红绸帷帐低垂",
        "喜堂": "宏伟古代府邸喜堂，红烛高烧，红毯铺地，仆从垂手肃立",
        "新房": "古代婚房，雕花拔步床，红帐低垂，烛光昏黄，香烟袅袅",
        "回廊": "深夜府邸长廊，灯笼光晕昏黄，两侧朱柱投下长影",
        "金钗": "烛光下的案几，鎏金首饰与密函堆叠",
        "惊醒": "深夜婚房床榻边，红帐摇曳，烛火剧烈晃动",
        "毒茶": "古代宗族正堂，香烟缭绕，匾额高悬，气氛凝滞",
        "吹灯": "深夜府邸幽暗回廊，红灯笼摇曳，阴影深重",
        "挡剑": "深夜婚房，红帐翻飞，烛光被风压得明灭不定",
        "拔剑": "深夜府邸内室，屏风绘着山水，灯笼光乱晃",
        "雪庭": "拂晓的古代庭院，积雪覆盖青石，红梅初绽，天色青灰",
    }
    for key, text in places.items():
        if place.startswith(key):
            return text
    return f"古代宅院{place}场景，烛火照明，夜色深沉"


def action_from(shot: dict) -> str:
    """Turn the English shot prompt into a Chinese action line.

    The English prompt describes the same beat, so the verbs and subjects are
    recovered by hand-tuned mapping keyed on the shot id rather than machine
    translation, which would mangle the period-drama register.
    """
    table = {
        "01_hairpin": "一名荆钗布裙的清秀年轻女子跪在暴雨中，一只侍女的粗糙手掌递来一支鎏金红穗金钗，特写金钗映着灯火",
        "02_mirror": "一名肤白唇红的美丽古装新娘对着铜镜，乌黑高髻上插着沉重的鎏金红穗金钗，一袭红绸盖头正缓缓覆下",
        "03_hall": "一名身着层叠红金明代婚服的孤独新娘缓步穿过宽阔的府邸喜堂，红毯长拖尾划过石地面，两侧仆从垂手鞠躬",
        "04_bride": "一名面色苍白的俊美古装青年男子昏迷在雕花拔步床上，身着暗红黑袍，一名红衣新娘坐在身旁缓缓掀起盖头，香烟袅袅",
        "05_corridor": "两名容貌一模一样的美丽古装女子在昏暗长廊中对峙，一人红衣婚服佩金钗，一人黑金刺绣华服冷笑，灯笼光横隔其间",
        "06_letter": "极近特写：纤细手指拧开中空的金钗，一卷细密信纸从中缓缓抽出，烛火在凤纹镌刻上流转",
        "07_wake": "一名苍白青年古装男子突然在床上睁眼，一把扣住女子的手腕，特写其漆黑眼底跳动的烛火，红帐翻卷",
        "08_tea": "一位着暗金华服的老妇人将一杯茶递给红衣新娘，一名留须中年男子在阴影中冷眼旁观，正堂香烟缭绕",
        "09_candle": "一名着黑金华服的身影吹灭廊下蜡烛，黑暗中银色刀光一闪，红灯笼剧烈摇晃",
        "10_sword": "一名红衣新娘扑身挡在苍白青年男子身前，银色刀刃刺穿她的肩膀，鲜血在红绸上晕开，特写",
        "11_draw": "青年男子从床榻上起身拔剑，鲜血溅上绘山水屏风，侍卫提灯笼涌入，低角度仰拍",
        "12_snow": "一名男子与一名女子并肩站在拂晓的积雪庭院中，她着红衣他着黑衣，各执半支断裂的金钗，雪花飘落，背影远景",
    }
    for key, text in table.items():
        if shot["id"].startswith(key):
            return text
    # Fall back to the original English content so nothing is silently dropped.
    return shot["prompt"]


def main() -> None:
    data = json.loads(SRC.read_text())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for shot in data["shots"]:
        camera = next((v for k, v in CAMERA.items()
                       if shot["id"].startswith(k)), "固定中景")
        # Single line on purpose: linenoiseNoTTY() splits stdin on '\n'
        # (linenoise.c:1620), so a multi-line prompt arrives as several lines
        # and every fragment before the last fires its own generation, which
        # corrupts the output ordering in h3_batch.py.
        prompt = (
            f"Scene: {scene_from(shot['title'])}。 "
            f"Action: {action_from(shot)}。 "
            f"Camera: {camera}。 "
            f"Look: {LOOK}。 "
            f"Audio: 环境声与烛火噼啪声，无配乐。"
        )
        lines.append(json.dumps({
            "id": shot["id"],
            "prompt": prompt,
            "out": f"{shot['id']}.mp4",
            "seed": 42,
        }, ensure_ascii=False))
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} shots -> {OUT}")
    for line in lines[:1]:
        print(json.loads(line)["prompt"])


if __name__ == "__main__":
    main()
