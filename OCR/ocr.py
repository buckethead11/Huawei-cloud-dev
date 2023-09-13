from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkocr.v1 import *


class HuaweiOCR:
    def __init__(self, ak, sk, region):
        """
        Initialize the HuaweiOCR class with authentication and region details.

        Parameters:
        ak (str): Access Key
        sk (str): Secret Key
        region (str): The data center region code
        """
        self.credentials = BasicCredentials(ak, sk)
        self.client = OcrClient.new_builder() \
            .with_credentials(self.credentials) \
            .with_region(OcrRegion.value_of(region)) \
            .build()

    def recognize_web_image(self, url):
        """
        Function to recognize a web image using Huawei OCR.

        Parameters:
        url (str): The URL of the web image.

        Returns:
        RecognizeWebImageResponse: The OCR response object.
        """
        request = RecognizeWebImageRequest()
        request.body = WebImageRequestBody(url=url)
        try:
            response = self.client.recognize_web_image(request)
            return response
        except exceptions.ClientRequestException as e:
            return {"status_code": e.status_code,
                    "request_id": e.request_id,
                    "error_code": e.error_code,
                    "error_msg": e.error_msg}

    @staticmethod
    def extract_words(response):
        """
        Function to extract words from Huawei OCR response and assemble them into sentences.

        Parameters:
        response (RecognizeWebImageResponse): The OCR response object.

        Returns:
        str: The assembled sentences.
        """
        words_list = []
        result = response.result
        words_block_list = result.words_block_list if result and hasattr(result, 'words_block_list') else []
        for block in words_block_list:
            word = block.words if hasattr(block, 'words') else ""
            words_list.append(word)
        sentence = " ".join(words_list)
        return sentence


if __name__ == "__main__":
    # Initialize HuaweiOCR class
    ocr_instance = HuaweiOCR("", "", "ap-southeast-2")
    
    # Recognize a web image and get the response
    ocr_response = ocr_instance.recognize_web_image(
        "https://signaturely.com/wp-content/uploads/2022/08/non-disclosure-agreement-uplead-791x1024.jpg")
    
    # Extract words from the response
    if isinstance(ocr_response, RecognizeWebImageResponse):
        sentence = HuaweiOCR.extract_words(ocr_response)
        print(sentence)
    else:
        print("An error occurred:", ocr_response)
