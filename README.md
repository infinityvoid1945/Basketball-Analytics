# Basketball-Analytics
This is a computer vision pipeline designed to track and analyze basketball gameplay. It can extract field goal percentage of a team at different areas of the court automatically, saving teams from manually reviewing clips.
After running the model, it will print a table in the console showing the field goal percentage from each area of the court.

# How this works
## Data

The model is trained on self-labeled data. Frames are pulled from basketball footage, pre-labeled by an earlier version of the model, then corrected by hand. It detects four classes, in this order: player, ball, rim, and backboard.

## Training

The detector is YOLO11x, fine-tuned from Ultralytics' pretrained weights.

Training uses cost-sensitive learning (some mistakes are punished more than others). In `train.py`, missing the ball when it's there (a false negative) costs 4× as much as missing an object of any other class. This addresses two imbalances: every frame has far more players than balls, and sharp balls are much more common than motion-blurred ones. Making the model more sensitive to missed balls overall puts most of the extra penalty on the blurred balls it was missing.

## Shot detection (`app.py`)

1. **Track the ball.** The tracker predicts the ball's next position from its velocity and gravity. A detection far from that prediction isn't trusted until a second one confirms it. If the ball briefly goes undetected, the tracker keeps following the predicted path.
2. **Judge the shot.** A shot counts as made if the ball goes above the top of the rim and then drops down through it. This is checked on the 2D camera image, so a ball passing behind the rim can look like a make.
3. **Find the zone.** The shooter (the last player holding the ball) has their foot position mapped onto a top-down court diagram using a homography. The homography is built from four court landmarks clicked on the first frame, so the camera has to stay still. The position is then sorted into a shot zone.

The output is an annotated video and a CSV of FG% by zone.

# Dependencies
To run this, you will need the following libraries downloaded in your IDE:
- opencv-python
- numpy
- ultralytics

# Model accuracy
Based on my own test cases, this is the accuracy: <br/>
<img width="962" height="175" alt="image" src="https://github.com/user-attachments/assets/709aaf4e-ab2b-41d6-acb6-3cd664dfb605" /> <br/>
True positive = shots detected made && shot made in the video <br/>
False positive = shots detected made && shot missed / did not take place in the video<br/>
True negative = shots detected missed && shot missed in the video<br/>
False negative = shots detected made && shot is made in the video<br/>

# How to use it
## 0. Preparing data
Before using this tool, you should film footages a basketball game. Note that you should film your video at a **fixed** angle and the entire three point arc should be visible.

## 1. Download model and code
Download both "cvxmodel.pt" and "app.py"

## 2. Modify path
Open app.py with an IDE. Edit MODEL_PATH by pasting where cvxmodel.pt is saved. For instance, if my cvxmodel.pt is saved at "C:\User\Downloads\cvxmodel.pt," then paste the directory into MODEL_PATH. Similarly, paste your footage path into app.py. <br/>
It would look something like this: <br/>
  MODEL_PATH = r"C:\Users\you\Downloads\cvxmodel.pt"
  VIDEO_PATH = r"C:\Users\you\Videos\game.mp4"
**If you are using paths with backslashes, please keep the r in front of the double quotes.**

## 3. Run app.py
Using your IDE, run app.py. If successful, a page asking you to label certain points would pop up. In my example, I have this:
<img width="1243" height="698" alt="image" src="https://github.com/user-attachments/assets/ee7d392f-a119-4f41-918c-00d0c4cc2ae7" />

As the console says, click the four following points:
   - LEFT corner-3 intersecting with baseline
   - RIGHT corner-3 intersecting with baseline
   - LEFT free-throw-line end
   - RIGHT free-throw-line end

The program will print in the console what you need to put into IMAGE_POINTS:
<img width="743" height="70" alt="image" src="https://github.com/user-attachments/assets/9c1208e3-42a9-4bf0-9966-b372556df6cf" />

Now, paste what you have in the console into your IMAGE_POINTS and make CALIBRATE_PICK **False**:
<img width="701" height="56" alt="image" src="https://github.com/user-attachments/assets/a8a2968e-aed8-401e-81af-66cea4eb14ed" />

After setting everything up, the model should start to analyze your footage and provide you a video, a csv file, and a table informing you about your shot accuracy. A sample would look like what I have below: <br/>
The video footage:<br/>
<img width="1280" height="720" alt="annotated_output3_1" src="https://github.com/user-attachments/assets/ae9d98f0-f071-4c72-bdbc-ed10a327cd1b" />


The table: <br/>
<img width="582" height="311" alt="屏幕截图 2026-06-23 101155" src="https://github.com/user-attachments/assets/8a737776-3019-463e-a294-b0aad2f98e9b" />


# Project timeline and future updates
This project started in December 2025, where I just started labelling data and discovered YOLO models.
<img width="1503" height="647" alt="image" src="https://github.com/user-attachments/assets/6890cb4e-aca5-4ca4-8941-f9220fdb9af2" />
For the next update, I am aiming towards improving the model's performance on dealing with edge cases (the model sometimes lose track of the ball when the ball is moving in high velocity) and making the model applicable when the camera is moving.

# Credit
## Thanks to everyone that guided me and community builders in Roboflow providing valuable data.
Thanks to user **Cricket** in Roboflow for making a dataset that contained basketball players. (https://universe.roboflow.com/cricket-qnb5l/basketball-xil7x/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true) 
