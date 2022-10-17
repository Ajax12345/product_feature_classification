import csv, re, typing
import os, numpy as np
import collections
import nltk.corpus, nltk.stem
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC


class Labels:
    def __init__(self, trans_func:typing.Callable = lambda x:x) -> None:
        self.trans_func = trans_func
        self.l_count = 0
        self.label_bindings = {}
        self.reverse_label_bindings = {}

    def __len__(self) -> int:
        return len(self.label_bindings)

    def __getitem__(self, label:str) -> int:
        if (_label:=self.trans_func(label)) not in self.label_bindings:
            self.label_bindings[_label] = self.l_count
            self.reverse_label_bindings[self.l_count] = _label
            self.l_count += 1
        
        return self.label_bindings[_label]

    def retrieve_label(self, label_id:int) -> str:
        return self.reverse_label_bindings[label_id]

        
def produce_file_rows(file_name:str, label_obj:Labels) -> typing.Iterator:
    with open(file_name) as f:
        c_num = 0
        for link, css_path, is_feature, text, feature_validation, customer_type, *_ in csv.reader(f):
            if c_num:
                if len(text) < 200:
                    if 'yes' in (l1:=feature_validation.lower()):
                        for feature in (re.findall('(?<=\()[^\)]+(?=\))', l1) or [l1]):
                            yield [text.lower(), label_obj[feature]]
                    else:
                        yield [text.lower(), label_obj['']]

            c_num += 1

def get_training_data(label_obj:Labels, folder='training_data') -> typing.Iterator:
    for i in os.listdir(t_path:=os.path.join(os.getcwd(), folder)):
        if i.endswith('.csv'):
            yield from produce_file_rows(os.path.join(t_path, i), label_obj)
    

def all_classifiers() -> list:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.experimental import enable_hist_gradient_boosting
    from sklearn.ensemble import HistGradientBoostingClassifier 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    return [SVC(gamma='auto'), GaussianNB(),HistGradientBoostingClassifier(),
                RandomForestClassifier(),LogisticRegression()]

def method_1():
    label_obj = Labels()
    n = [*get_training_data(label_obj)]
    vectorizer = CountVectorizer()
    bag = vectorizer.fit_transform(np.array([a for a, _ in n]))
    X = bag.toarray()
    scaler = preprocessing.StandardScaler().fit(X)
    X = scaler.transform(X)
    y = np.array([b for _, b in n])
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs', multi_class='ovr', max_iter=1000)
    lr.fit(X_train, y_train)
    y_predict = lr.predict(X_test)
    print("LogisticRegression Accuracy %.3f" %metrics.accuracy_score(y_test, y_predict))

def method_2():
    #running accuracy: 0.903303509979353
    label_obj = Labels()
    n = [*get_training_data(label_obj)]
    label1 = Labels(lambda x:x.lower())
    _X = [[label1[j] for j in re.findall('\w+', a)] for a, _ in n]
    m_l = max(map(len, _X))
    X = [i+[-1]*(m_l - len(i)) for i in _X]
    y = [b for _, b in n]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))
    

def method_3():
    #0.7076923076923077 (~ 0.65 - 0.70)
    label_obj, feat_type = Labels(), collections.defaultdict(list)
    n = [*get_training_data(label_obj)]
    for a, b in n:
        feat_type[bool(label_obj.retrieve_label(b))].append([a, b])
    
    
    label1 = Labels(lambda x:x.lower())
    x_sample = feat_type[1]+feat_type[0][:len(feat_type[1])*2]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    _X = [[label1[lemmatizer.lemmatize(j)] for j in re.findall('\w+', a) if j not in nltk.corpus.stopwords.words('english')] for a, _ in x_sample]
    m_l = max(map(len, _X))
    X = [i+[-1]*(m_l - len(i)) for i in _X]
    y = [b for _, b in x_sample]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))

def method_4():
    #accuracy: 0.6923076923076923
    label_obj, feat_type = Labels(), collections.defaultdict(list)
    n = [*get_training_data(label_obj)]
    for a, b in n:
        feat_type[bool(label_obj.retrieve_label(b))].append([a, b])

    x_sample = feat_type[1]+feat_type[0][:len(feat_type[1])*2]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    _X = [' '.join(lemmatizer.lemmatize(j) for j in re.findall('[a-zA-Z\-]+', a) if j not in nltk.corpus.stopwords.words('english')) for a, _ in x_sample]
    cv = CountVectorizer()
    tf_transformer = TfidfTransformer()
    X = cv.fit_transform(_X).toarray()
    X = tf_transformer.fit_transform(X).toarray()
    y = [b for _, b in x_sample]
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    for clf in all_classifiers():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print(clf, metrics.accuracy_score(y_test, y_pred))
    
    '''
    SVC(gamma='auto') 0.6520710059171597
    GaussianNB() 0.4390532544378698
    HistGradientBoostingClassifier() 0.6319526627218935
    RandomForestClassifier() 0.7183431952662722
    LogisticRegression() 0.6710059171597633
    '''

    '''
    clf = SVC(gamma='auto')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(metrics.accuracy_score(y_test, y_pred))
    '''

if __name__ == '__main__':
    '''
    Questions: 
    - what is the overall quality of our data?
    - how to best Vectorize input? (currently using `Labels` object, had errors with sklearn's CountVectorizer)
    - what should the ratio of feature/non_feature be? (Currently 1126/10498) 
    - perhaps ignore all training data that has score < 0.5, unless it has been specificially labeled?
    - is there a better classifier type to use for this problem i.e RandomForrest?
    - how should the input text best be preprocessed?

    TODO: 
    - Run full training set against a new site's text
    - Remember that a single feature description/bit of text can have multiple paraphrases, need to factor that into accuracy calculator
    '''
    method_4()
    
    