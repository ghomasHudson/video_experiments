from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import PIL.Image
import pytesseract
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from datasets_multimodal import tokens_to_frames
import easyocr
import cv2


def calculate_ocr_losses(batch, predictions, targets, vocab=None, limit=999999):
    b, t, c, x, y = predictions.size()
    input_text = []
    target_text = []
    input_text_with_vocab = []
    for i in range(min(b, limit)):
        ocr_text = get_text_from_video(predictions[i], method="google")
        input_text.append(ocr_text)
        target_text.append(batch["config"][i]["output"])

        if vocab is not None:
            #ocr_text_with_vocab = get_text_from_video(predictions[i], vocab=vocab)
            with_vocab = []
            for word in ocr_text.split():
                with_vocab.append(get_closest_word(word, vocab))
            ocr_text_with_vocab = " ".join(with_vocab)
            #ocr_text_with_vocab = get_closest_word(ocr_text, vocab)
            input_text_with_vocab.append(ocr_text_with_vocab)

    ocr_stats = {}
    ocr_stats['ocr_accuracy'] = accuracy_score(target_text, input_text)
    ocr_stats['ocr_f1'] = f1_score(target_text, input_text, average='weighted')
    ocr_stats['ocr_precision'] = precision_score(target_text, input_text, average='weighted', zero_division=0)
    ocr_stats['ocr_recall'] = recall_score(target_text, input_text, average='weighted', zero_division=0)

    if vocab is not None:
        ocr_stats['ocr_accuracy_with_vocab'] = accuracy_score(target_text, input_text_with_vocab)
        ocr_stats['ocr_f1_with_vocab'] = f1_score(target_text, input_text_with_vocab, average='weighted')
        ocr_stats['ocr_precision_with_vocab'] = precision_score(target_text, input_text_with_vocab, average='weighted', zero_division=0)
        ocr_stats['ocr_recall_with_vocab'] = recall_score(target_text, input_text_with_vocab, average='weighted', zero_division=0)

    return ocr_stats



reader = easyocr.Reader(['en'], gpu=False)
def easy_ocr_frame_to_text(frame):
    """
    Convert a frame to text
    """
    result = reader.readtext(frame, detail=0)
    return " ".join(result)

# def distance(s1, s2):
#     '''Hamming distance between two strings'''
#     return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def distance(s1, s2):
    '''Edit distance between two strings'''
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def get_closest_word(word, vocab):
    '''Get the closest word in the vocab list'''
    min_dist = float('inf')

    closest_word = None
    for v in vocab:
        dist = distance(word, v)
        if dist < min_dist:
            min_dist = dist
            closest_word = v
    return closest_word

def process_frame(frame, size=None):
    is_tensor = isinstance(frame, torch.Tensor)

    # Upscale the image
    if size is not None:
        if is_tensor:
            frame = to_pil_image(frame)
        frame = frame.resize((size, size), PIL.Image.BICUBIC)
        if is_tensor:
            frame = pil_to_tensor(frame)
    if is_tensor:
        #frame = frame[0].cpu().float().numpy()
        frame = frame.cpu().float().numpy()
    if np.max(frame) <= 1:
        frame = frame * 255

    return frame.astype(np.uint8)


def get_text_from_video(frames, vocab=None, method="google"):
    """
    Get the text from the frame
    """
    t, c, h, w = frames.shape

    if vocab is not None:
        vocab = vocab + ["<pad>", ","]

    if method == "reference":
        if vocab is None:
            raise ValueError("Vocab must be provided when using reference method")

        # Generate the frames from the tokens
        reference_frames = {}
        for token in vocab:
            frame = tokens_to_frames([token], frame_size=512, img_type="RGB")[0].numpy()
            reference_frames[token] = frame




    # Perform OCR on the image
    texts = []
    texts_with_vocab = []
    for i in range(t):
        pred_frame = process_frame(frames[i], size=512)
        match method:
            case "google":
                from google.oauth2 import service_account
                from google.cloud import vision
                from io import BytesIO
                from PIL import Image
                credentials = service_account.Credentials.from_service_account_file(
                    filename="client.json",
                    scopes=["https://www.googleapis.com/auth/cloud-platform"]
                )
                client = vision.ImageAnnotatorClient(credentials=credentials)
                im = Image.fromarray(pred_frame)
                buffer = BytesIO()
                im.save(buffer, format="PNG")
                image = vision.Image(content=buffer.getvalue())
                response = client.text_detection(image=image)
                if len(response.text_annotations) == 0:
                    pred = ""
                else:
                    pred = response.text_annotations[0].description
            case "easyocr":
                pred = reader.readtext(pred_frame, detail=0)
                pred =  " ".join(result).strip()
            case "doctr":
                from doctr.io import DocumentFile
                from doctr.models import ocr_predictor
                model = ocr_predictor("parseq", pretrained=True)
                doc = DocumentFile.from_images(["test/pred.png"])
                pred = model(doc)
            case "tesseract":
                pred = pytesseract.image_to_string(pred_frame).strip()
                psm = 3
                while len(pred) == 0:
                    try:
                        print("Trying again with psm={}".format(psm))
                        pred = pytesseract.image_to_string(pred_frame, config='--psm {}'.format(psm)).strip()
                        psm += 1
                    except:
                        pred = ""
            case "reference":
                closest_match = None
                closest_distance = float("inf")
                for sample_label in reference_frames:
                    sample_image = reference_frames[sample_label]
                    diff = cv2.absdiff(pred_frame, sample_image)
                    distance = diff.sum()
                    if distance < closest_distance:
                        closest_distance = distance
                        closest_match = sample_label
                pred = closest_match
            case _:
                raise ValueError("Invalid method")


        if pred == "<pad>":
            pred = ""

        if vocab is not None and pred != "":
            pred_with_vocab = get_closest_word(pred, vocab)
            if pred_with_vocab == "<pad>":
                pred_with_vocab = ""
            texts_with_vocab.append(pred_with_vocab)

        texts.append(pred)

    output =  {
        "text": " ".join(texts).strip(),
        "text_with_vocab": " ".join(texts_with_vocab).strip()
    }
    return output


if __name__ == '__main__':
    '''
    import easyocr
    reader = easyocr.Reader(['en'])
    result = reader.readtext('./test/pred.png', detail=0)
    results = " ".join(result)
    print(result)
    import sys; sys.exit(0)
    #text = pytesseract.image_to_string(PIL.Image.open('ocr.png'))
    '''
    img_pred = PIL.Image.open('test/pred.png')
    img_pred = pil_to_tensor(img_pred).unsqueeze(0)
    # img_tgt = PIL.Image.open('test/true.png')
    # img_tgt = pil_to_tensor(img_tgt).unsqueeze(0)

    metrics = ocr_text_metrics(img_pred, img_tgt, debug=True, vocab=['dog', 'apple', 'frog', 'cat'])
    print(metrics)
    #text = get_text_from_video(img_pred, method="google", vocab=['dog', 'apple', 'frog', 'cat'])
    print(text)
