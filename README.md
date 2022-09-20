# Hackathon
----------------My coach-----------------

The program supposes to serve the peoples how wants to have feedback on their sport exercises performance.
The program follows the movements of the trainee and gives him feedback on the correctness of his movements.

At this moment the program supports two exercises:
1 - left serratus stretch
2 - lift weights (with both hands)
Navigation between exercises will be by pressing on the appropriate number, i.e. for the left serratus stretch exercise the user will click on the '1' button (it's also the default exercise)
and for the lift weights exercise the user will click on the '2' button.

In each exercise, the program trace after users hands, and by calculating the degrees of user's hands parts the program know
where his hands refer to the exercise stages.

In each exercise we defined two stages the user must go through:
The first one is the initiate hands position before the user start the exercise (or after he has done a one-time exercise he supposes to return his hands there)
The second is the end hands position in the exercise, when the user gets there he should return his hands to the first position and repeat the exercise over again how much time he wants.

When we start the program a window opens and reads the frames from the user's web camera. 
In the window we can see (except to the web-camera video) in the upper left corner two things:
1 - the instructions for the user, what should he do with his hands (or other relevant body parts). for example, in the second exercise (lift weights) the user needs to start with his hand's
downside therefore it will be written UP (means the user needs to move up his hands). And when he will, the text will change to DOWN ((means the user needs to move up his hands) and so on.

2 - in the second row will be written "succeeded: (number of successes)", i.e. each time the user will do the exercise perfectly the successes number will be increased by 1.


---------------------------------
The code was written as part of the hackathon, therefore the code is a little bit messy and incomplete due to the shortness of time.

for any comments on the code, you can send an email to: yohananyakir@gmail.com, Omerdayan94@gmail.com
