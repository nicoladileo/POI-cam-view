""" 
Copyright (C) 2015  Nicola Dileo
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
Module: cam.py
--------
"""

import cv2
import sys



if __name__ == '__main__':
	cascPath = 'haarcascade_frontalface_default.xml'
	faceCascade = cv2.CascadeClassifier(cascPath)
	args = sys.argv
	if len(args) > 1:
		targ_name = sys.argv[1]
	else:
		targ_name = "admin.png"
	targ_img = cv2.imread(targ_name, -1)
	
	print('$- Start in %s mode'%(targ_name.replace(".png","").upper()))
	video_capture = cv2.VideoCapture(0)

	while True:
	    ret, frame = video_capture.read()

	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.2,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.cv.CV_HAAR_SCALE_IMAGE
	    )

	    for (x, y, w, h) in faces:
		x_offset=x
		y_offset=y
		new_targ = cv2.resize(targ_img,(int(w*1.1),int(h*1.1)))
	    	for c in range(0,3):
	    		frame[y_offset:y_offset+new_targ.shape[0], x_offset:x_offset+new_targ.shape[1], c] = new_targ[:,:,c] * (new_targ[:,:,3]/255.0) +  frame[y_offset:y_offset+new_targ.shape[0],x_offset:x_offset+new_targ.shape[1], c] * (1.0 - new_targ[:,:,3]/255.0)
	
	    
	    
	    cv2.imshow('Video', frame)

	    if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	video_capture.release()
	print('$- Bye')
	cv2.destroyAllWindows()
	

