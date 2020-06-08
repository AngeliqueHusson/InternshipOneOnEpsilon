#!C:\Users\s157165\Anaconda3\envs\InternshipOneOnEpsilon\python.exe
from youtube_transcript_api import YouTubeTranscriptApi
import requests
import cgi, cgitb
cgitb.enable()
import string
import pickle
from nltk.corpus import stopwords, words
from nltk.corpus import words
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model


class Classification:
    def website(method, file, title):
        # category_id_df = pd.read_csv("category_id_df.csv")
        # id_to_category = dict(category_id_df[['y_id', 'newHashtag']].values)  # Dictionary connecting id to hashtag
        id_to_category = {0: 'Exploding Dots', 1: 'Statistics and probability', 2: 'Geometry', 3: 'Numbers',
                          4: 'Polygons lines and quadrilaterals', 5: 'Arithmetic', 6: 'Algebra',
                          7: 'Logic and discrete maths', 8: 'Measurement', 9: 'Trigonometry', 10: 'Calculus'}

        # Porter stemming method
        ps = PorterStemmer()
        My_dict = {}
        for i in words.words():
            My_dict[i] = i

        # hash table implemtation of dictionary
        textString = file.replace('/n', "")
        textString1 = title.replace('/n', "")

        # eliminate the punctuation in form of characters
        textString = [char for char in textString if char not in string.punctuation]
        textString1 = [char for char in textString1 if char not in string.punctuation]
        textString = ''.join(textString)
        textString1 = ''.join(textString1)
        textString = textString.lower()
        textString1 = textString1.lower()

        # tokenize
        textString_token = word_tokenize(textString)
        textString_token1 = word_tokenize(textString1)

        # stopwords eliminating 1st time
        textString_stop = [word for word in textString_token if word not in stopwords.words('english')]
        textString_stop1 = [word for word in textString_token1 if word not in stopwords.words('english')]

        # Spelling checking and dividing two connected words
        list1 = []
        for word in textString_stop:
            if word in My_dict:
                list1.append(word)
            else:
                list2 = []
                count_j = 0
                for j in range(1, len(word)):
                    if word[:j] in My_dict and word[j:] in My_dict:
                        list2.append(word[:j])
                        list2.append(word[j:])
                        count_j += 1
                if count_j == 1:
                    list1.extend(list2)

        # eliminate stopwords 2nd time
        clean_sentence = [word for word in list1 if word.lower() not in stopwords.words('english')]
        clean_title = [word for word in textString_stop1 if word.lower() not in stopwords.words('english')]

        qw = len(clean_sentence)
        tp = 20
        qx = len(clean_title)
        tl = tp * qw / (100 - tp)
        m = round(tl / qx)

        for j in range(1, m):
            clean_sentence.append(title)  # Adding the titles to the captions

        ## Stemming
        text1 = []
        for w in clean_sentence:
            x = ps.stem(w)  # Stemming method
            text1.append(x)

        text = ' '.join(text1)
        text = str(text)
        global proba
        global label2
        global proba2

        if method == "Support Vector Machine":
            loaded_model = pickle.load(open('C:/Users/s157165\Documents/Jaar 5 2019-2020 Master/Internship Australia/MathClassification/cgi-bin/finalized_model_SVM.sav', 'rb'))
            vectorizer1 = pickle.load(open('C:/Users/s157165\Documents/Jaar 5 2019-2020 Master/Internship Australia/MathClassification/cgi-bin/vectorizer1.sav','rb'))
            result = loaded_model.predict(vectorizer1.transform([text]))
            result = id_to_category[result[0]]
        elif method == "Recurrent Neural Network":
            # Load model
            loaded_model = load_model('C:/Users/s157165\Documents/Jaar 5 2019-2020 Master/Internship Australia/MathClassification/cgi-bin/finalized_model_RNN.h5')

            #Load tokenizer
            tokenizer = Tokenizer()
            with open('C:/Users/s157165\Documents/Jaar 5 2019-2020 Master/Internship Australia/MathClassification/cgi-bin/tokenizerRNN.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)

            # text_labels = encoder.classes_
            text = [str(text1)]

            # tokenizer.fit_on_texts(text)
            X_test = tokenizer.texts_to_sequences(text)
            text = pad_sequences(X_test, maxlen=500)

            result = loaded_model.predict_classes(text)
            result = id_to_category[result[0]]

            proba1 = max(loaded_model.predict_proba(text))
            proba = round(max(proba1)*100)
            proba1 = list(proba1)
            # print(proba1)
            proba2 = sorted(proba1)[9]  # The second highest value
            label2 = proba1.index(proba2)  # Index of the second highest value
            label2 = id_to_category[label2]  # Corresponding category belonging to this second highest value
            proba2 = round(proba2*100)
        else:
            loaded_model = pickle.load(open('C:/Users/s157165\Documents/Jaar 5 2019-2020 Master/Internship Australia/MathClassification/cgi-bin/finalized_model_LR.sav', 'rb'))
            vectorizer1 = pickle.load(open('C:/Users/s157165\Documents/Jaar 5 2019-2020 Master/Internship Australia/MathClassification/cgi-bin/vectorizer1.sav', 'rb'))
            result = loaded_model.predict(vectorizer1.transform([text]))
            result = id_to_category[result[0]]

            proba1 = max(loaded_model.predict_proba(vectorizer1.transform([text])))
            proba = round(max(proba1)*100)
            proba1 = list(proba1)
            # print(proba1)
            proba2 = sorted(proba1)[9]  # The second highest value
            label2 = proba1.index(proba2)  # Index of the second highest value
            label2 = id_to_category[label2]  # Corresponding category belonging to this second highest value
            proba2 = round(proba2*100)

        return result

# Get input fields
form = cgi.FieldStorage()
complete = True

if "videoID" in form:
    videoID = form.getvalue("videoID")
else:
    complete = False

if "dropdown" in form:
   subject = form.getvalue('dropdown')
else:
   complete = False

# If you want to run the python file outside of the server environment, comment out the code before, and use the lines below
# videoID = "riXcZT2ICjA"
# complete = True
# subject = "Recurrent Neural Network"

# Video ID obtained from url
if complete == False:
    print("Content-type: text/html\n\n")
    print("<p id=\"red\"> Please enter all fields and try again</p>")
else:
    # Obtain video captions
    try:
        caption = YouTubeTranscriptApi.get_transcript(video_id=videoID, languages=['en'])
        caption1 = [i['text'] for i in caption]
        textVideo = ''.join(caption1)
    except:
        textVideo = " "


    ### Key.txt includes my YouTube API key
    file = open("C:/Users/s157165\Documents/Jaar 5 2019-2020 Master/Internship Australia/MathClassification/cgi-bin/Key.txt")
    key = (file.read()).rstrip("\n")

    # Set DEVELOPER_KEY to the API key value from the APIs & auth > Registered apps
    # https://cloud.google.com/console
    DEVELOPER_KEY = key
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"

    url = "https://www.googleapis.com/youtube/v3/videos?part=snippet&id={id}&key={api_key}"
    r = requests.get(url.format(id=videoID, api_key=DEVELOPER_KEY))
    data = r.json()

    titleVideo = data['items'][0]['snippet']['title']

    webs = Classification.website(subject, textVideo, titleVideo)

    print("Content-type: text/html\n\n")

    # print("Content-Type: text/html\r\n\r\n")
    # print("<html>")
    # print("<head><title>CGI program</title></head>")
    # print("<body>")
    print("<h1>Classification results</h1>")
    # print("<p>The title of the video:</p>")
    # print(titleVideo)
    # print("<p>The text of the video:</p>")
    # print(textVideo)
    print("<p>The video called \" %s \"  with video ID  \"%s\" is classified as: \" <a id=\"bold\" > %s  </a> \" using classification algorithm %s.</p>" % (titleVideo, videoID, webs, subject))
    if subject != "Support Vector Machine":
        p = "%"
        print("<p> The %s algorithm is %s %s sure that this is the correct label.</p> " % (subject,proba,p))
        print("<p></p>")
        print("<p> If you do not agree with this result, it might be interesting to take a look at the second highest ranked label. The second highest ranked label is category \"%s\", the %s algorithm is %s %s sure that this is the correct label. </p>" % (label2, subject, proba2, p))
    # print("</body>")
    # print("</html>")