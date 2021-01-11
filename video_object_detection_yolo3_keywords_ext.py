

# Detecting Objects on Video with OpenCV deep learning library
#
# Algorithm:
# Reading input video --> Loading YOLO v3 Network -->
# --> Reading frames in the loop --> Getting blob from the frame -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
# --> Writing processed frames
#
# Result:
# New video file with Detected Objects, Bounding Boxes and Labels


# Importing needed libraries
import numpy as np
import cv2
import time


# Keywords extraction

from collections import OrderedDict
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

import en_core_web_lg
nlp = en_core_web_lg.load()



import nltk
from pprint import pprint
import pandas as pd
from nltk.tokenize import sent_tokenize


class TextRank4Keyword():
    """Extract keywords from text"""
    
    def __init__(self):
        self.d = 0.85 # damping coefficient, usually is .85
        self.min_diff = 1e-5 # convergence threshold
        self.steps = 10 # iteration steps
        self.node_weight = None # save keywords and its weight

    
    def set_stopwords(self, stopwords):  
        """Set stop words"""
        for word in STOP_WORDS.union(set(stopwords)):
            lexeme = nlp.vocab[word]
            lexeme.is_stop = True
    
    def sentence_segment(self, doc, candidate_pos, lower):
        """Store those words only in cadidate_pos"""
        sentences = []
        for sent in doc.sents:
            selected_words = []
            for token in sent:
                # Store words only with cadidate POS tag
                if token.pos_ in candidate_pos and token.is_stop is False:
                    if lower is True:
                        selected_words.append(token.text.lower())
                    else:
                        selected_words.append(token.text)
            sentences.append(selected_words)
        return sentences
        
    def get_vocab(self, sentences):
        """Get all tokens"""
        vocab = OrderedDict()
        i = 0
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = i
                    i += 1
        return vocab
    
    def get_token_pairs(self, window_size, sentences):
        """Build token_pairs from windows in sentences"""
        token_pairs = list()
        for sentence in sentences:
            for i, word in enumerate(sentence):
                for j in range(i+1, i+window_size):
                    if j >= len(sentence):
                        break
                    pair = (word, sentence[j])
                    if pair not in token_pairs:
                        token_pairs.append(pair)
        return token_pairs
        
    def symmetrize(self, a):
        return a + a.T - np.diag(a.diagonal())
    
    def get_matrix(self, vocab, token_pairs):
        """Get normalized matrix"""
        # Build matrix
        vocab_size = len(vocab)
        g = np.zeros((vocab_size, vocab_size), dtype='float')
        for word1, word2 in token_pairs:
            i, j = vocab[word1], vocab[word2]
            g[i][j] = 1
            
        # Get Symmeric matrix
        g = self.symmetrize(g)
        
        # Normalize matrix by column
        norm = np.sum(g, axis=0)
        g_norm = np.divide(g, norm, where=norm!=0) # this is ignore the 0 element in norm
        
        return g_norm

    
    def get_keywords(self, number=10):

        keyword_key = ""
    

        """Print top number keywords"""
        node_weight = OrderedDict(sorted(self.node_weight.items(), key=lambda t: t[1], reverse=True))
        for i, (key, value) in enumerate(node_weight.items()):
            print(key + ' - ' + str(value))
            
            keyword_key = str(keyword_key) + "," + str(key)
            if i > number:
                break
        
        print(keyword_key)
        return keyword_key
        
        
    def analyze(self, text, 
                candidate_pos=['NOUN', 'PROPN'], 
                window_size=4, lower=False, stopwords=list()):
        """Main function to analyze text"""
        
        # Set stop words
        self.set_stopwords(stopwords)
        
        # Pare text by spaCy
        doc = nlp(text)
        
        # Filter sentences
        sentences = self.sentence_segment(doc, candidate_pos, lower) # list of list of words
        
        # Build vocabulary
        vocab = self.get_vocab(sentences)
        
        # Get token_pairs from windows
        token_pairs = self.get_token_pairs(window_size, sentences)
        
        # Get normalized matrix
        g = self.get_matrix(vocab, token_pairs)
        
        # Initionlization for weight(pagerank value)
        pr = np.array([1] * len(vocab))
        
        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr = (1-self.d) + self.d * np.dot(g, pr)
            if abs(previous_pr - sum(pr))  < self.min_diff:
                break
            else:
                previous_pr = sum(pr)

        # Get weight for each node
        node_weight = dict()
        for word, index in vocab.items():
            node_weight[word] = pr[index]
        
        self.node_weight = node_weight




def keywordsExtraction(text) :

    tr4w = TextRank4Keyword()
    tr4w.analyze(text, candidate_pos = ['NOUN', 'PROPN'], window_size=4, lower=False)
    
    keyword_key = ""

    """Print top number keywords"""
    node_weight = OrderedDict(sorted(tr4w.node_weight.items(), key=lambda t: t[1], reverse=True))
    list_keyword_key = []
    for i, (key, value) in enumerate(node_weight.items()):
        #print(key + ' - ' + str(value))
        
        # keyword_key = str(keyword_key) + "," + str(key)
        list_keyword_key.append(str(key))
        
        if i > 30:
            break
    
    
    
    #print(list_keyword_key)
    return list_keyword_key


# input scenairo! >>> 외부 문서에서 불러올 것!
# scenairo = """ """
scenairo = open('script/Script.txt', 'r')

# keywords Extract! 리스트 형태로 출력, 각 값을 이미지 인식 레이블과 비교할 것, 이 결과를 가지로 인식되는 이미지의 레이블과 비교해도 되고 아래의 명사추출 결과만을 가지고 비교해도 됨
keywordsExtraction(scenairo)

print("keyword extraction:", keywordsExtraction(scenairo))




# 사니리오에서 주요 명사만 추출한다면,
def build_dictionary(text_):
    
    essay_input_corpus = str(text_) #문장입력
    essay_input_corpus = essay_input_corpus.lower()#소문자 변환

    sentences  = sent_tokenize(essay_input_corpus) #문장 토큰화
    dictionary = {}
    for sent in sentences:
        pos_tags = nltk.pos_tag(nltk.word_tokenize(sent))
        for tag in pos_tags:
            value = tag[0]
            pos = tag[1]
            dictionary[value] = pos
    return dictionary

pos_dict = build_dictionary(scenairo)

nouns = [n for n, tag in pos_dict.items() if tag in ["NN","NNP"] ]
print("시나리오에서 주요명사 추출",nouns) # 이 결과값(리스트 형식으로 저장됨)을 이미지 인식 레이블과 비교하여, 인식되는 프레임의 시간값을 추출하여 별도 저장하여 편집구간을 찾아낼 것! 

"""
Start of:
Reading input video
"""

# Defining 'VideoCapture' object
# and reading video from a file
# Pay attention! If you're using Windows, the path might looks like:
# r'videos\traffic-cars.mp4'
# or:
# 'videos\\traffic-cars.mp4'
video = cv2.VideoCapture('/Users/kimkwangil/Documents/VISION2020/김윤하_영상AI/movie/Scence1/DSCF0319.MP4')

# Preparing variable for writer
# that we will use to write processed frames
writer = None

# Preparing variables for spatial dimensions of the frames
h, w = None, None

"""
End of:
Reading input video
"""


"""
Start of:
Loading YOLO v3 network
"""

# Loading COCO class labels from file
# Opening file
# Pay attention! If you're using Windows, yours path might looks like:
# r'yolo-coco-data\coco.names'
# or:
# 'yolo-coco-data\\coco.names'
with open('yolo-coco-data/coco.names') as f:
    # Getting labels reading every line
    # and putting them into the list
    labels = [line.strip() for line in f]


# # Check point
print('List with labels names:')
print(labels)

# Loading trained YOLO v3 Objects Detector
# with the help of 'dnn' library from OpenCV
# Pay attention! If you're using Windows, yours paths might look like:
# r'yolo-coco-data\yolov3.cfg'
# r'yolo-coco-data\yolov3.weights'
# or:
# 'yolo-coco-data\\yolov3.cfg'
# 'yolo-coco-data\\yolov3.weights'
network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()
print("layers_names_all", layers_names_all)
# # Check point
# print()
# print(layers_names_all)

# Getting only output layers' names that we need from YOLO v3 algorithm
# with function that returns indexes of layers with unconnected outputs
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# # Check point
# print()
print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Check point
# print()
print(type(colours))  # <class 'numpy.ndarray'>
print(colours.shape)  # (80, 3)
print(colours[0])  # [172  10 127]

"""
End of:
Loading YOLO v3 network
"""


"""
Start of:
Reading frames in the loop
"""

# Defining variable for counting frames
# At the end we will show total amount of processed frames
f = 0

# Defining variable for counting total time
# At the end we will show time spent for processing all frames
t = 0

# Defining loop for catching frames
while True:
    # Capturing frame-by-frame
    ret, frame = video.read()

    # If the frame was not retrieved
    # e.g.: at the end of the video,
    # then we break the loop
    if not ret:
        break

    # Getting spatial dimensions of the frame
    # we do it only once from the very beginning
    # all other frames have the same dimension
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]

    """
    Start of:
    Getting blob from current frame
    """

    # Getting blob from current frame
    # The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob from current
    # frame after mean subtraction, normalizing, and RB channels swapping
    # Resulted shape has number of frames, number of channels, width and height
    # E.G.:
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    """
    End of:
    Getting blob from current frame
    """

    """
    Start of:
    Implementing Forward pass
    """

    # Implementing forward pass with our blob and only through output layers
    # Calculating at the same time, needed time for forward pass
    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    print("output_from_network:", output_from_network)
    end = time.time()

    # Increasing counters for frames and total time
    f += 1
    t += end - start

    # Showing spent time for single current frame
    print('Frame number {0} took {1:.5f} seconds'.format(f, end - start))

    """
    End of:
    Implementing Forward pass
    """

    """
    Start of:
    Getting bounding boxes
    """

    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []
    
    # Going through all output layers after feed forward pass
    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]

            # # Check point
            # # Every 'detected_objects' numpy array has first 4 numbers with
            # # bounding box coordinates and rest 80 with probabilities
            #  # for every class
            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original frame and in this way get coordinates for center
                # of bounding box, its width and height for original frame
                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)

    """
    End of:
    Getting bounding boxes
    """

    """
    Start of:
    Non-maximum suppression
    """

    # Implementing non-maximum suppression of given bounding boxes
    # With this technique we exclude some of bounding boxes if their
    # corresponding confidences are low or there is another
    # bounding box for this region with higher confidence

    # It is needed to make sure that data type of the boxes is 'int'
    # and data type of the confidences is 'float'
    # https://github.com/opencv/opencv/issues/12789
    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

    """
    End of:
    Non-maximum suppression
    """

    """
    Start of:
    Drawing bounding boxes and labels
    """

    # Checking if there is at least one detected object
    # after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding box
            # and converting from numpy array to list
            colour_box_current = colours[class_numbers[i]].tolist()

            # # # Check point
            print(type(colour_box_current))  # <class 'list'>
            print(colour_box_current)  # [172 , 10, 127]

            # Drawing bounding box on the original current frame
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)



            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], # >>>>>>>>>>> 여기의 레이블과 사니라오의 토픽이 같으면, 시간을 기록하거나 화면에 편집구간 표시할 것
                                                   confidences[i])
            print("text_box_current::::::::::", labels[int(class_numbers[i])])

            label_name = labels[int(class_numbers[i])] #save detected label of img

#####################  작성해야 할 코드 #############################################                
            # label_name 과 추출한 nouns 값이 같으면, 그 프레임의 시간을 txt파일로 저장한다.


            # Putting text with label and confidence on the original image
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)

    """
    End of:
    Drawing bounding boxes and labels
    """

    """
    Start of:
    Writing processed frame into the file
    """

    # Initializing writer
    # we do it only once from the very beginning
    # when we get spatial dimensions of the frames
    if writer is None:
        # Constructing code of the codec
        # to be used in the function VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # Writing current processed frame into the video file
        # Pay attention! If you're using Windows, yours path might looks like:
        # r'videos\result-traffic-cars.mp4'
        # or:
        # 'videos\\result-traffic-cars.mp4'
        writer = cv2.VideoWriter('result_video/result-Scence1_0319.mp4', fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    # Write processed current frame to the file
    writer.write(frame)

    """
    End of:
    Writing processed frame into the file
    """

"""
End of:
Reading frames in the loop
"""


# Printing final results
print()
print('Total number of frames', f)
print('Total amount of time {:.5f} seconds'.format(t))
print('FPS:', round((f / t), 1))


# Releasing video reader and writer
video.release()
writer.release()