import streamlit as st
import easyocr
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Streamlitページ設定
st.set_page_config(page_title="習字サポート", layout="centered")
st.title('Shuji')
st.write('習字の練習をサポートします。')
st.write('一文字だけアップロードしてください。')

# 画像から文字をトリミングする。
def find_largest_bbox(contours):
    if not contours:  # 輪郭が検出されなかった場合
        return None

    # 全ての輪郭を包含する最小の矩形を見つける
    min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + w)
        max_y = max(max_y, y + h)

    # 整数にキャストする
    return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

def crop_and_resize_character(image, new_size=(300, 300)):
    # PILのImageオブジェクトをNumPy配列に変換
    if isinstance(image, Image.Image):
        image = np.array(image)

    # OpenCVを使用して画像をグレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 二値化処理
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # ノイズ除去
    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)

    # 輪郭検出
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 最大の外接矩形を見つける
    bbox = find_largest_bbox(contours)
    if bbox:
        x, y, w, h = bbox
        # トリミング
        cropped_image = image[y:y+h, x:x+w]
    else:
        # 輪郭が見つからなかった場合は元の画像を使用
        cropped_image = image

    # 画像のサイズを変更
    resized_image = cv2.resize(cropped_image, new_size, interpolation=cv2.INTER_AREA)

    return resized_image

def add_text_to_image(image, text, font_path='玉ねぎ楷書激無料版v7改.ttf'):
    # オリジナル画像のサイズを取得
    image_size = image.size

    # フォントサイズを調整
    font_size = image_size[1] - 30

    # テキスト画像の生成
    text_image = Image.new("RGBA", image_size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_image)

    # フォントの読み込み
    font = ImageFont.truetype(font_path, font_size)

    # テキストのバウンディングボックスを計算
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # テキストを中央に配置
    x = (image_size[0] - text_width) / 2
    y = (image_size[1] - text_height) / 2
    
    # テキストの描画
    draw.text((x, y), text, fill=(255, 0, 0, 110), font=font)

    # オリジナル画像の上にテキスト画像を重ねる
    combined_image = Image.alpha_composite(image.convert("RGBA"), text_image)

    return combined_image

# ファイルアップロード
uploaded_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

# ストリームリットアプリケーションのメイン部分
if uploaded_file is not None:
    # PILを使用して画像を読み込む
    image = Image.open(BytesIO(uploaded_file.getvalue()))

    # 画像をリサイズしてファイルサイズを減らす
    base_width = 600
    w_percent = (base_width / float(image.size[0]))
    h_size = int((float(image.size[1]) * float(w_percent)))
    image = image.resize((base_width, h_size), Image.Resampling.LANCZOS)

    # 画像をJPEG形式に変換してさらにサイズを軽減
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    image = Image.open(buffered)
    
    # EasyOCRリーダーの初期化
    reader = easyocr.Reader(['ja'])

    # OCR処理
    image_for_ocr = np.array(image)  # PIL ImageをNumPy配列に変換
    results = reader.readtext(image_for_ocr)

    if len(results) >= 2:
        st.error('一文字だけで再度アップロードしてください。')
    else:
        # 結果を表示
        text = ""
        for result in results:
            text += result[1] + "\n"
        st.write(text)
        
        # トリミングとリサイズ
        resized_image = crop_and_resize_character(image)  # PIL Imageオブジェクトを渡す
        resized_image = Image.fromarray(resized_image)    # NumPy配列をPIL Imageに変換

        # お手本テキストの追加
        final_image = add_text_to_image(resized_image, text)

        # 処理した画像を表示
        st.image(final_image, caption='Processed Image', use_column_width=True)
