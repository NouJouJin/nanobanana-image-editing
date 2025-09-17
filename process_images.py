import google.generativeai as genai
from PIL import Image
import os
import io
from dotenv import load_dotenv

# --- ユーザー設定エリア ---

# 1. 編集指示を【日本語】で記述
# 例1 (参考画像あり): "入力画像の人物に、参考画像のTシャツを着せてください。背景はそのままにしてください。"
# 例2 (文字入れ): "入力画像の中央に、スタイリッシュなフォントで 'SUMMER SALE' という文字を追加してください。"
# 例3 (スタイル適用): "入力画像の風景を、参考画像のような幻想的な油絵のスタイルに変更してください。"
# 例4 (参考画像あり): "入力画像の端末の裏側のロゴデザインを参考画像のロゴを当てはめて、右上のドローンも自然な形で表現してください"
# 例5: "入力画像に写っている左の人物を男性に置き換えよろしくお願いいたします。"
PROMPT_TEXT_JP = "入力画像の左の男性について、農家に伴走する人物らしい、農家の衣装を着た人物にして表現してください"

# 2. 入力・出力・参考画像のフォルダ名
INPUT_DIR = "input_images"
OUTPUT_DIR = "output_images"
REFERENCE_DIR = "reference_images" # 参考画像フォルダ

# 3. 使用する参考画像のファイル名を指定 (reference_imagesフォルダ内にあるファイル名)
#    参考画像を使わない場合は、"" (空文字列) にしてください。
# 例: REFERENCE_IMAGE_NAME = "tshirt_design.png"
# 例: REFERENCE_IMAGE_NAME = "style_van_gogh.jpg"
# 例: REFERENCE_IMAGE_NAME = "PersonStyle.png" 
REFERENCE_IMAGE_NAME = "LOGO.png" 

# --- スクリプト本体 ---

def setup_directories():
    for dir_path in [INPUT_DIR, OUTPUT_DIR, REFERENCE_DIR]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def translate_prompt_to_english(text_model, japanese_prompt):
    """【Proモデル使用】日本語プロンプトを英語に翻訳する関数"""
    print("--- プロンプトの自動翻訳を開始 (by gemini-1.5-pro) ---")
    print(f"原文 (日本語): {japanese_prompt}")
    
    translation_instruction = (
        "Please translate the following Japanese text into natural, high-quality English "
        "for an image generation AI. Only output the translated text itself, with no extra explanations.\n\n"
        f"Japanese: '{japanese_prompt}'\n\n"
        "English:"
    )
    
    try:
        response = text_model.generate_content(translation_instruction)
        translated_text = response.text.strip()
        if translated_text:
            print(f"翻訳後 (英語): {translated_text}")
            print("-------------------------------------------------")
            return translated_text
        else:
            print("❌ 翻訳結果が空です。")
            return None
    except Exception as e:
        print(f"❌ 翻訳中にエラーが発生しました: {e}")
        return None

def process_image(image_model, image_path, prompt, reference_image=None):
    """画像を処理する関数"""
    try:
        print(f"🔄 処理中: {os.path.basename(image_path)}")
        
        input_img = Image.open(image_path)
        
        contents = [prompt, input_img]
        if reference_image:
            contents.append(reference_image)
            print(f"   参考画像: '{REFERENCE_IMAGE_NAME}' を使用します。")
        
        response = image_model.generate_content(contents)
        
        if not response.candidates:
            print(f"❌ エラー: {os.path.basename(image_path)} のレスポンスに生成候補がありませんでした。")
            if response.prompt_feedback:
                print(f"   APIからのフィードバック: Block Reason = {response.prompt_feedback.block_reason}")
            return

        image_part = next((part for part in response.candidates[0].content.parts if part.inline_data), None)

        if not image_part:
            print(f"❌ エラー: {os.path.basename(image_path)} のレスポンスに画像データが含まれていませんでした。")
            return

        image_data = image_part.inline_data.data
        result_img = Image.open(io.BytesIO(image_data))
        
        print(f"   生成された画像のサイズ: {result_img.size[0]} x {result_img.size[1]}")
        
        base_name, ext = os.path.splitext(os.path.basename(image_path))
        output_path = os.path.join(OUTPUT_DIR, f"{base_name}_edited.png")
        
        result_img.save(output_path)
        print(f"✅ 完了: {output_path} に保存しました。")

    except Exception as e:
        print(f"❌ エラー発生 ({os.path.basename(image_path)}): {e}")

def main():
    print("--- Gemini画像処理スクリプト開始 ---")
    
    load_dotenv()
    setup_directories()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: .envファイルにGOOGLE_API_KEYを設定してください。")
        return
        
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        print(f"APIキーの設定でエラーが発生しました: {e}")
        return

    print("モデルを初期化しています...")
    try:
        text_model = genai.GenerativeModel('gemini-1.5-pro-latest')
        # ユーザーの指定通り、モデル名を維持
        image_model = genai.GenerativeModel('gemini-2.5-flash-image-preview')
        print("モデルの初期化完了。")
    except Exception as e:
        print(f"モデルの初期化中にエラーが発生しました: {e}")
        return
    
    translated_prompt = translate_prompt_to_english(text_model, PROMPT_TEXT_JP)
    if not translated_prompt:
        print("プロンプトの翻訳に失敗したため、処理を終了します。")
        return
        
    reference_image_obj = None
    if REFERENCE_IMAGE_NAME:
        ref_path = os.path.join(REFERENCE_DIR, REFERENCE_IMAGE_NAME)
        if os.path.exists(ref_path):
            reference_image_obj = Image.open(ref_path)
        else:
            print(f"エラー: 参考画像 '{ref_path}' が見つかりません。")
            return

    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"'{INPUT_DIR}' フォルダに処理対象の画像がありません。")
        return

    print(f"\n{len(image_files)} 件の画像を処理します。")
    
    for filename in image_files:
        image_path = os.path.join(INPUT_DIR, filename)
        process_image(image_model, image_path, translated_prompt, reference_image=reference_image_obj)
        print("-" * 20)
        
    print("--- 全ての処理が終了しました ---")

if __name__ == "__main__":
    main()