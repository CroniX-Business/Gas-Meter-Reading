# Gas Meter Reading using YOLOv8 and PaddleOCR

This project utilizes machine learning techniques to read gas meters automatically. The process involves training a YOLOv8 model to detect the portion of the meter containing the numbers, followed by extracting this portion and passing it through PaddleOCR for digit recognition. The entire system is wrapped in a Streamlit web application for easy usability.

If you want to run app on streamlit you need to add this project to your github and then pass that github on streamlit web.
I had some problems with compactibility of some libraries between streamlit web and paddleOCR.

1. Once the application is running, upload an image containing a gas meter.
2. The YOLOv8 model will detect the region of interest containing the numbers on the meter.
3. The detected portion will be cropped and passed through PaddleOCR for digit recognition.
4. The recognized digits will be displayed on the app interface.
   
## Contributing
Contributions are welcome! If you'd like to improve this project, please feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the developers of YOLOv8 and PaddleOCR for their fantastic work and to the Streamlit team for making the creation of web applications simple and intuitive.
Special thank to my friend who worked with me on this project for college project

Feel free to adjust the content according to your specific project details and preferences.
