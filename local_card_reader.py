import cv2
import numpy as np
import pytesseract
from PIL import Image

def get_suit_from_color(img_bgr):
    """Determine suit based on dominant color in the image."""
    # Convert to HSV
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Define color ranges
    # Note: Carbon deck might be different, tuning needed.
    # Standard:
    # Red (Hearts)
    # Green (Clubs)
    # Blue (Diamonds)
    # Black (Spades)

    # We can just sum RGB pixels if we know the background is card white/gray
    # Calculate average color of the center (where the suit is)
    mean_color = cv2.mean(img_bgr)[:3] # B, G, R
    b, g, r = mean_color

    # Logic for PokerStars 4-Color Deck
    # Clubs = Green (High G)
    # Diamonds = Blue (High B)
    # Hearts = Red (High R)
    # Spades = Black (Low RGB)

    if g > r + 20 and g > b + 20: return 'c'
    if b > r + 20 and b > g + 20: return 'd'
    if r > g + 20 and r > b + 20: return 'h'
    return 's' # Default to spade/black

def recognize_card(pil_img):
    """Recognize a single card (Rank + Suit)."""
    # Convert to CV2
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]

    # 1. Rank (Top Left Corner)
    # Crop top 40%, left 40%
    rank_img = img_cv[0:int(h*0.4), 0:int(w*0.4)]

    # Preprocess for OCR
    gray = cv2.cvtColor(rank_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # Config: Single char, whitelist
    cfg = "--psm 10 -c tessedit_char_whitelist=23456789TJQKA10"
    rank_text = pytesseract.image_to_string(thresh, config=cfg).strip()

    # Clean up
    rank_map = {'1': 'T', '0': 'O', '|': '1'} # Common errors
    rank = rank_text.upper()
    rank = rank_map.get(rank, rank)

    # Fallback: Count pixels? (A, K, Q are complex, 2, 3 simple)
    # For now trust OCR
    if not rank: return None

    # 2. Suit (Color Analysis)
    # Crop bottom right or center
    # Suit is usually below rank or in center
    suit_img = img_cv[int(h*0.4):, :]
    suit = get_suit_from_color(suit_img)

    return f"{rank}{suit}"

def recognize_cards_local(pil_image, is_board=False):
    """
    Detect cards in a card area (Hero hand or Board).
    Returns list of strings e.g. ['Ah', 'Td']
    """
    # 1. Find contours of cards?
    # Or assume fixed positions?
    # Simple approach: Threshold > Find Contours > Filter by Aspect Ratio

    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cards = []
    # Sort contours by x position
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[0])

    for i, cnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(cnt)

        # Filter noise
        if w < 20 or h < 30: continue

        # Aspect ratio check (Cards are vertical rectangles)
        ratio = h / w
        if 1.2 < ratio < 1.6:
            # Found a card candidate
            # Pad slightly
            card_crop = pil_image.crop((x, y, x+w, y+h))

            # DEBUG: Save crop
            try:
                card_crop.save(f"debug_images/cards/detected_{i}.png")
            except: pass

            res = recognize_card(card_crop)
            if res:
                print(f"DEBUG: Found Card {res} at x={x}")
                cards.append(res)
            else:
                print(f"DEBUG: Failed to recognize card at x={x}")

    return cards

def read_pot_local(pil_img):
    """Read total pot from board image using OCR."""
    # Convert to CV2
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # Preprocess: Invert (Text is usually white on dark)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    # OCR
    text = pytesseract.image_to_string(thresh, config='--psm 6').replace('\n', ' ')

    # Parse "Total Pot: 12.5 BB"
    # Find float number
    import re
    matches = re.findall(r"[\d\.]+", text)
    if matches:
        try:
            # Return largest number found (likely pot)
            vals = [float(m) for m in matches if m != "."]
            return max(vals) if vals else 0.0
        except:
            pass
    return 0.0
