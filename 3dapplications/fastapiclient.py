import os
import io
import cv2
import requests
import numpy as np
# import matplotlib.image as mpimg
# import cv2
# from IPython.display import Image, display


# Some example images
image_files = ["apple.jpg", "clock.jpg", "oranges.jpg", "car.jpg"]
from PIL import Image
# for image_file in image_files:
#     print(f"\nDisplaying image: {image_file}")
#     img=cv2.imread(f"images/{image_file}")
#     # display(Image(filename=f"images/{image_file}"))

#     plt.imshow(img)
#     plt.axis('off')
#     plt.show()

# for image_file in image_files:
#     print(f"\nDisplaying image: {image_file}")
#     img=Image.open(f"images/{image_file}")
#     # display(Image(filename=f"images/{image_file}"))

#     # plt.imshow(img)
#     # plt.axis('off')
#     # plt.show()
#     img.show()


def response_from_server(url, image_file, verbose=True):
    """Makes a POST request to the server and returns the response.

    Args:
        url (str): URL that the request is sent to.
        image_file (_io.BufferedReader): File to upload, should be an image.
        verbose (bool): True if the status of the response should be printed. False otherwise.

    Returns:
        requests.models.Response: Response from the server.
    """

    files = {"file": image_file}
    response = requests.post(url, files=files)
    status_code = response.status_code
    print(response.status_code)
    if verbose:
        msg = (
            "Everything went well!"
            if status_code == 200
            else "There was an error when handling the request."
        )
        print(msg)
    return response


base_url = "http://localhost:8000"
endpoint = "/predict"
model = "yolov3-tiny"

url_with_endpoint_no_params = base_url + endpoint
print(url_with_endpoint_no_params)

full_url = url_with_endpoint_no_params + "?model=" + model
print(full_url)

with open("images/clock2.jpg", "rb") as image_file:
    prediction = response_from_server(full_url, image_file)

dir_name = "images_predicted"
if not os.path.exists(dir_name):
    os.mkdir(dir_name)


def display_image_from_response(response):
    """Display image within server's response.

    Args:
        response (requests.models.Response): The response from the server after object detection.
    """

    image_stream = io.BytesIO(response.content)
    print("here is the response")
    print(image_stream)
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    filename = "image_with_objects.jpeg"
    cv2.imwrite(f"images_predicted/{filename}", image)
    img = Image.open(f"images_predicted/{filename}")
    img.show()
    # display(Image(f'images_predicted/{filename}'))


display_image_from_response(prediction)

image_files = ["car2.jpg", "clock3.jpg", "apples.jpg"]

for image_file in image_files:
    with open(f"images/{image_file}", "rb") as image_file:
        prediction = response_from_server(full_url, image_file, verbose=False)

    display_image_from_response(prediction)
