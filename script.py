import pandas
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def read_file(file_name):
    return pandas.read_csv(file_name)

def label_encoder(file):
    encoder = LabelEncoder()
    for column in file.columns:
        file[column] = encoder.fit_transform(file[column])
    return file


def train_and_test_set(x, y):
    return train_test_split(x, y, random_state=30)


def separate_set(file):
    # csv files consist of 14 columns, last one being the type of disease,
    return file.iloc[:, 0:9], file.iloc[:, 13]


def logistic_regression(x_train, x_test, y_train, y_test):
    print("Logistic regression:")

    #liblinear
    lr_model = LogisticRegression().fit(x_train, y_train)
    y_predicted = lr_model.predict(x_test)
    print("\t Logistic regression (liblinear solver) score: " + str(metrics.f1_score(y_test, y_predicted, average='micro')))

    #newton-cg
    lr_model = LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(x_train, y_train)
    y_predicted = lr_model.predict(x_test)
    print("\t Logistic regression (multinomial, newton-cg solver) score: " + str(metrics.f1_score(y_test, y_predicted, average='micro')))

    print("*******************************************")

def support_vector_machine(x_train, x_test, y_train, y_test):

    print("SVM:")

    #Linear
    svm_model = SVC(kernel='linear').fit(x_train, y_train)
    y_predicted = svm_model.predict(x_test)
    print("\t Linear score: " + str(metrics.f1_score(y_test, y_predicted, average='micro')))

    #RBF
    svm_model = SVC(kernel='rbf').fit(x_train, y_train)
    y_predicted = svm_model.predict(x_test)
    print("\t RBF score: " + str(metrics.f1_score(y_test, y_predicted, average='micro')))

    print("*******************************************")


def naive_bayes(x_train, x_test, y_train, y_test):
    print("NAIVE BAYES")

    # Gaussian
    nb_model = GaussianNB().fit(x_train, y_train)
    y_prob = nb_model.predict(x_test)
    print("\tGaussian naive bayes score: " + str(metrics.f1_score(y_test, y_prob, average='micro')))

    # Bernoulli
    nb_model = BernoulliNB().fit(x_train, y_train)
    y_prob = nb_model.predict(x_test)
    print("\tBernoulli naive bayes score: " + str(metrics.f1_score(y_test, y_prob, average='micro')))

    # Multinomial
    nb_model = MultinomialNB().fit(x_train, y_train)
    y_prob = nb_model.predict(x_test)
    print("\tMultinomial naive bayes score: " + str(metrics.f1_score(y_test, y_prob, average='micro')))

    print("*******************************************")

def k_neighbors_classifier(x_train, x_test, y_train, y_test):
    print("K NEIGHBORS CLASSIFIER:")

    # Euclidean
    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean').fit(x_train, y_train)
    y_predicted = knn.predict(x_test)
    print("\t Euclidean metrics score: " + str(metrics.f1_score(y_test, y_predicted, average='micro')))

    # Manhattan
    knn = KNeighborsClassifier(n_neighbors=4, metric='manhattan').fit(x_train, y_train)
    y_predicted = knn.predict(x_test)
    print("\t Manhattan metrics score: " + str(metrics.f1_score(y_test, y_predicted, average='micro')))

    # Chebyshev
    knn = KNeighborsClassifier(n_neighbors=3, metric='chebyshev').fit(x_train, y_train)
    y_predicted = knn.predict(x_test)
    print("\t Chebyshev neighbors score: " + str(metrics.f1_score(y_test, y_predicted, average='micro')))

    print("*******************************************")


def analyze_database(file_path, db_name):
    data = label_encoder(read_file(file_path))
    x, y = separate_set(data)
    print(db_name);
    print("\nType of disease recognition scores\n")
    x_train, x_test, y_train, y_test = train_and_test_set(x, y)
    support_vector_machine(x_train, x_test, y_train, y_test) # SVM
    naive_bayes(x_train, x_test, y_train, y_test) # Naive Bayes
    k_neighbors_classifier(x_train, x_test, y_train, y_test) # K Neighbors
    logistic_regression(x_train, x_test, y_train, y_test) # Logistic regression

    print("\nIs disease present recognition scores\n")
    y = numpy.where(y >= 1, 1, 0)
    x_train, x_test, y_train, y_test = train_and_test_set(x, y)
    support_vector_machine(x_train, x_test, y_train, y_test) # SVM
    naive_bayes(x_train, x_test, y_train, y_test) # Naive Bayes
    k_neighbors_classifier(x_train, x_test, y_train, y_test) # K Neighbors
    logistic_regression(x_train, x_test, y_train, y_test) # Logistic regression



# Main
if __name__ == '__main__':
    #cleveland
    analyze_database('data/cleveland.csv', "CLEVELAND DATABASE")
    #switzerland
    analyze_database('data/switzerland.csv', "SWITZERLAND DATABASE")
    #va
    analyze_database('data/va.csv', "VA DATABASE")
    #hungarian
    analyze_database('data/hungarian.csv', "HUNGARIAN DATABASE")
