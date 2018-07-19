# single_shot_face_recognition
  This is intended to slolve the classic problem in facial recognition called one_sample_per_person problem.
  Human have various facial expressions. Each of these expressions are mostly influenced by 'orbicularis oris' which is situated just below eyes and connecting to upper region of upper lip.
  Size and intensity in the expression is mostly dependant on 'Bucccinator' or 'whistling muscle' which is situated on side ways of the cheeks connecting to the ends of lips.
  All these muscles are mostly visible when the axis of head is parallel to that of neck, this leaves us with a great option to train faces on this smaple.
  Then we generate a 128 bit encoding to each faces where we use knn to find closest faces. This model is tolerant to faces which are shifted to a degree of 45 degrees with 94% accuracy and till 60 degrees with 68% accuraracy.
  
  Device specifications :
  Python 3 and above
  scipy
  scikit-learn
  scikit-image
  flask
  numpy
  pandas
  dlib
  face_recognition modules.
  
  
  Model can be trained on your own images by modifying facae_recognition_knn.py
