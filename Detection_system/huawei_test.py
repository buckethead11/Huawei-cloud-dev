from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdkimage.v2 import ImageClient, RunImageTaggingRequest, ImageTaggingReq
from huaweicloudsdkimage.v2.region.image_region import ImageRegion

if __name__ == "__main__":
    ak = "AB39NJWJBQGMLXKSX7JX"
    sk = "CxGQm3DtEas0sE0C9XvlLTHeY8Q9AVFsPdOn7Nq9"

    credentials = BasicCredentials(ak, sk)

    client = ImageClient.new_builder() \
        .with_credentials(credentials) \
        .with_region(ImageRegion.value_of("ap-southeast-1")) \
        .build()

    try:
        request = RunImageTaggingRequest()
        request.body = ImageTaggingReq(
            limit=50,
            threshold=75,
            language="en",
            url="https://i.ibb.co/P1cjPtw/image.png"
        )
        response = client.run_image_tagging(request)

        if response.status_code == 200:
            print("Response Content:")
            print(response)

        else:
            print("Image tagging request failed with status code:", response.status_code)

    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)