<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />



# Paara ft. Swayam


## Basic Details
### Team Name: Clock'in


### Team Members
- Team Lead: Rabahuddin K P - Government Engineering College, Kozhikode
- Member 2: Fathima Fidha MA - Government Engineering College, Kozhikode

### Project Description
Paara is a student sleepiness detection system that uses a webcam to check whether a student is becoming sleepy.

It detects things like closed eyes, head leaning down, resting the head on the desk, and yawning. The system runs directly in the browser, so the user only needs to open the website and allow camera access.

### The Problem (that doesn't exist)
Student sleeping in class who needs to be woken up by a meme audio targeted at them

### The Solution (that nobody asked for)
We built 'Paara' to look for different signs of sleepiness:

- Eyes staying closed: Detects how long the eyes remain closed, ignoring normal blinking, and identifies it as sleeping when they stay closed for too long.

- Head staying down: detects when students head remains down for too long.

- Head resting on the desk: detects when students head is close to desk for too long

- Yawning: Detects when student Yawns

After detecting, a meme audio is played aloud to wake up and humorously target the sleeping student

For example, a normal blink should not be detected as sleeping. So the eyes need to stay closed for a few seconds before the system marks the student as sleeping.

## Technical Details
### Technologies/Components Used
For Software:
- **HTML**
- **CSS**
- **JavaScript**
- **Python** - used for the first working prototype
- **OpenCV** - used during the prototype
- **MediaPipe** - face and body landmark detection
- **Flask** - used in the first prototype


For Hardware:
- Laptop
- Built-in webcam

### Implementation
For Software:
I built the project in two main stages.
First Prototype,
I first made a working version using Python, OpenCV, MediaPipe and Flask.
The webcam was read by Python and the MediaPipe models were used to detect the face and body.
I started by testing each part separately:
After getting each part working, I combined them to detect sleepiness.
Detecting Closed Eyes,
To detect closed eyes, I used the points around the eyes provided by MediaPipe.
I calculated the Eye Aspect Ratio (EAR).
EAR = Eye height / Eye width


# Screenshots (Add at least 3)
<img width="1280" height="832" alt="Screenshot 2026-09-12 at 6 45 53 AM" src="https://github.com/user-attachments/assets/b7090714-5aa6-4457-b9fb-6347744a1443" />
*Eyes open*

<img width="1280" height="832" alt="Screenshot 2026-09-12 at 6 48 33 AM" src="https://github.com/user-attachments/assets/46f5bec9-9210-4336-bb3d-58389c7edd38" />
*Eyes Closed*

<img width="1280" height="832" alt="Screenshot 2026-09-12 at 6 50 18 AM" src="https://github.com/user-attachments/assets/1e028f64-0647-4640-b625-7db0ca9c09ac" />
*Eyes closed for 5 sec detects sleeping*

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)



