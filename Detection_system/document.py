from PIL import Image
import pytesseract
import cv2

# Set the path to the Tesseract executable (you may need to adjust this)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Open the image of the document
image_path = 'document2.jpg'
image = cv2.imread(image_path)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to enhance contrast
threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Perform denoising
denoised_image = cv2.fastNlMeansDenoising(threshold_image, None, 10, 7, 21)

# Convert OpenCV image to PIL image
pil_image = Image.fromarray(denoised_image)

# Use pytesseract to extract text from the processed image
text = pytesseract.image_to_string(pil_image)

print(text)