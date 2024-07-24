import numpy
import sys
from sklearn.ensemble import RandomForestClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import numpy as np
from numpy import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.tree import export_graphviz
from sklearn import tree

numpy.set_printoptions(threshold=sys.maxsize) #print entire numpy arrays

def create_target_vector_for_rf_model(student_semester_data, student_next_semester_data, course_subject, course_number):
    """

    creating a target vector (0 if the student is not in the course and 1 if the student is) for rf model

    :param student_semester_data: complete df of student data for a semester (have not turned it into a feature matrix yet)
    :param student_next_semester_data: complete df of student data for the semester that follows the semester of student_semester_data
    :param course_subject: the subject of the course we are predicting students to be in or not
    :param course_number: the number of the course we are predicting students to be in or not
    :return: the target vector where 1 indicates the student is in the given course the following semester or 0 if they are not
    """
    target_vector = np.zeros(shape=(len(student_semester_data), 1))

    for i in range(0, len(student_semester_data)):
        if student_semester_data.iloc[i]["SPRIDEN_PIDM"] in student_next_semester_data["SPRIDEN_PIDM"].values:  # if student attended FSU the next semester
            target_vector[i] = int(student_next_semester_data[
                                           student_next_semester_data["SPRIDEN_PIDM"] == student_semester_data.iloc[i][
                                               "SPRIDEN_PIDM"]][course_subject + "_" + course_number].values) # copy value from column that indicates student enrollment in course

        else:  # if student didn't attend FSU the next semester, they didn't take the course
            target_vector[i] = 0

    return target_vector

def create_features_matrix_for_rf_model(semester_data):
    """

    creating features matrix of student data for rf model

    :param semester_data: semester of students whose information we are using to create a features matrix for random forest model
    :return: student info feature matrix
    """
    semester_data.drop(columns=semester_data.iloc[:, :3], inplace=True)
    semester_data = semester_data.iloc[:, :-1]

    return semester_data


def create_rf_model_for_course(all_student_data, course_subject, course_number):
    """

    creating a random forest (rf) model using spring student data to predict whether a student is in a given course
    in a fall semester

    :param all_student_data: df of all student data obtained from student_data_without_majors_edited.xlsx
    :param course_subject: course subject of course we are predicting enrollment for
    :param course_number: course number of course we are predicting enrollment for
    :return: N/A
    """

    spring_2021_students_df = all_student_data.loc[
        (all_student_data["Academic Term"] == "Spring") & (all_student_data["Academic Year"] == "2020-2021")]

    fall_2021_students_df = all_student_data.loc[
        (all_student_data["Academic Term"] == "Fall") & (all_student_data["Academic Year"] == "2021-2022")]

    target_vector = create_target_vector_for_rf_model(spring_2021_students_df, fall_2021_students_df, course_subject, course_number)
    spring_2021_students_df = create_features_matrix_for_rf_model(spring_2021_students_df)

    random.seed(1234) #create random seed to allow replicability of model
    rf_model = BalancedRandomForestClassifier(random_state=random.seed(1234), class_weight="balanced_subsample") # BalancedRandomForest() provides each tree with a balanced subsample
                                                                                                                 # where there are a balanced amount of majority and minority class observations
                                                                                                                 # class_weight="balanced_subsample" adjusts weights of the majority/minority classes
    rf_model.fit(spring_2021_students_df, target_vector)
    test = dict(zip(spring_2021_students_df.columns, rf_model.feature_importances_))
    test = sorted(test.items(), key=lambda x: x[1], reverse=True)
    print(test)
    # target_vector_columns = pd.Index(["0","1"])
    # tree.plot_tree(rf_model.estimators_[0], feature_names=spring_2021_students_df.columns, class_names=target_vector_columns, filled=True)
    # plt.show()

    spring_2022_students_df = all_student_data.loc[
        (all_student_data["Academic Term"] == "Spring") & (all_student_data["Academic Year"] == "2021-2022")]

    fall_2022_students_df = all_student_data.loc[
        (all_student_data["Academic Term"] == "Fall") & (all_student_data["Academic Year"] == "2022-2023")]

    target_vector = create_target_vector_for_rf_model(spring_2022_students_df, fall_2022_students_df, course_subject, course_number)
    spring_2022_students_df = create_features_matrix_for_rf_model(spring_2022_students_df)

    y_pred = rf_model.predict(spring_2022_students_df)

    # fpr, tpr, threshold = metrics.roc_curve(y_pred, target_vector) # the ROC curve indicates performance of the model at various probabalistic thresholds
    # roc_auc = metrics.auc(fpr, tpr)
    # plt.title('ROC Curve')
    # plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    # plt.legend(loc='lower right')
    # plt.plot([0, 1], [0, 1], 'r--')
    # plt.xlim([0, 1])
    # plt.ylim([0, 1])
    # plt.ylabel('True Positive Rate')
    # plt.xlabel('False Positive Rate')
    # plt.show()

    cm = confusion_matrix(target_vector, y_pred) # the confusion matrix of a classification model shows the amount of
                                                 # true positives, false positives, true negatives, and false negatives
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    cm_display.plot()
    cm_display.figure_.set()
    plt.show()
    tn, fp, fn, tp = cm.ravel() # obtain true positives and others from confusion matrix
    precision = tp / (tp + fp) # precision asks: out of all my predictions, how accurate have I been in predicting the positive class?
    recall = tp / (tp + fn) # recall asks out of all the positive cases in the sample, how many of these have been correctly identified?
    print("precision:", precision)
    print("recall:", recall)

    spring_2022_students_df = all_student_data.loc[ # create df again so that all columns are there in upcoming prints
        (all_student_data["Academic Term"] == "Spring") & (all_student_data["Academic Year"] == "2021-2022")]

    print(f"\n\nstudents in {course_subject} {course_number}") #students from the previous semester that are actually in the course the next semester
    for i in range(0, len(target_vector)):
        if target_vector[i] == 1:
            print(spring_2022_students_df.iloc[i].to_frame().T.to_string()) # student in course

    print(f"\nstudents predicted to be in {course_subject} {course_number}")
    for i in range(0, len(target_vector)):
        if y_pred[i] == 1:
            print(spring_2022_students_df.iloc[i].to_frame().T.to_string())

    print(f"{int(sum(target_vector))} students from spring 2022 took the course in fall 2022. {tp} predictions were correct. there were {fp} false positives and {fn} false negatives.")

def main():
    # this file is basically just making the random forest using new data set where student data includes the major code. seems to be worse than other data set without major codes?

    # student_data = pd.read_excel("student_data_with_majors_edited.xlsx")
    # pd.to_pickle(student_data, "student_data_with_majors_edited.pkl")

    student_data = pd.read_pickle("student_data_with_majors_edited.pkl")
    student_data = pd.get_dummies(student_data,  # do one-hot encoding on the following columns
                                  columns=['SGBSTDN_COLL_CODE_1', 'SGBSTDN_COLL_CODE_2', 'SGBSTDN_MAJR_CODE_1',
                                           'SGBSTDN_MAJR_CODE_2'], dtype=int)
    create_rf_model_for_course(student_data, "BSBA", "2209")

if __name__ == "__main__":
    main()