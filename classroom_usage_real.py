import datetime
import sys
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, r2_score
from sklearn.preprocessing import OrdinalEncoder
numpy.set_printoptions(threshold=sys.maxsize) #print entire numpy arrays
sns.set(font_scale=1.5) #set graph font size

# import warnings
# warnings.filterwarnings("ignore")
# pd.options.mode.chained_assignment = None

def export_course_statistics_to_xlsx(classes_data, old_classes_data):
    """
    method to export section count, enrollment count, actual enrollment ratio, and predicted enrollment ratios using various techniques

    :param classes_data: data obtained from Course Data Set 6-26.xlsx
    :param old_classes_data: data obtained from Course Dataset for Summer Program June20.xlsx
    :return:
    """
    sections_counts_by_class = classes_data.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                     "Academic Term"])[["CRS Section Number"]].nunique()

    enrollment_counts_by_class = classes_data.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                       "Academic Term"])[["Enrollment"]].sum()

    sections_counts_by_class["Number Enrolled"] = enrollment_counts_by_class  # adding enrollment by course

    enrollment_by_year_and_term = old_classes_data.groupby(["Academic Year", "Academic Term"])[
        ["SPRIDEN_PIDM"]].nunique()  # counting enrollment by year and term

    number_enrolled_div_by_total_term_enrollment_ratio = np.zeros(shape=(13826, 1))  # create array to store ratio

    for i, index in enumerate(
            sections_counts_by_class.index):  # divide course enrollment by total enrollment for that year and term
        number_enrolled_div_by_total_term_enrollment_ratio[i] = (sections_counts_by_class["Number Enrolled"].iloc[i] /
                                                                 int(enrollment_by_year_and_term.loc[(
                                                                     sections_counts_by_class.iloc[i].name[2],
                                                                     sections_counts_by_class.iloc[i].name[3])]))

    sections_counts_by_class["Enrollment Ratio"] = number_enrolled_div_by_total_term_enrollment_ratio

    sections_counts_by_class_copy = sections_counts_by_class.copy()  # make copy to use when exporting to excel
                                                                     # makes excel file easier to read

    sections_counts_by_class.reset_index(inplace=True)  # reset index to use indices for filling array of predictions

    prediction_list_using_all_years_and_lin_reg = np.zeros(shape=(13826, 1))
    prediction_list_using_three_years_and_lin_reg = np.zeros(shape=(13826, 1))
    prediction_list_using_all_years_and_average = np.zeros(shape=(13826, 1))
    prediction_list_using_3_years_and_average = np.zeros(shape=(13826, 1))
    prediction_list_using_1_year_and_average = np.zeros(shape=(13826, 1))

    for i, course in enumerate(sections_counts_by_class.groupby(["CRS Subject", "CRS Course Number", "Academic Term"])
                               ["Enrollment Ratio"].unique().index): # create predictions using each technique

        predictions, indices_for_sheet = lin_reg_for_enrollment_ratio(course, sections_counts_by_class,
                                                                      "all")  # run a model for each course
        for i, index in enumerate(indices_for_sheet):  # populate predictions array
            prediction_list_using_all_years_and_lin_reg[index] = predictions[i]

        predictions, indices_for_sheet = lin_reg_for_enrollment_ratio(course, sections_counts_by_class,
                                                                      "3")  # run a model for each course
        for i, index in enumerate(indices_for_sheet):
            prediction_list_using_three_years_and_lin_reg[index] = predictions[i]

        predictions, indices_for_sheet = average_for_enrollment_ratio(course, sections_counts_by_class,
                                                                      "all")  # run a model for each course
        for i, index in enumerate(indices_for_sheet):
            prediction_list_using_all_years_and_average[index] = predictions[i]

        predictions, indices_for_sheet = average_for_enrollment_ratio(course, sections_counts_by_class,
                                                                      "3")  # run a model for each course
        for i, index in enumerate(indices_for_sheet):
            prediction_list_using_3_years_and_average[index] = predictions[i]

        predictions, indices_for_sheet = average_for_enrollment_ratio(course, sections_counts_by_class,
                                                                      "1")  # run a model for each course
        for i, index in enumerate(indices_for_sheet):
            prediction_list_using_1_year_and_average[index] = predictions[i]

    sections_counts_by_class_copy[
        "Ratio Pred. from Lin. Reg. Using All Data"] = prediction_list_using_all_years_and_lin_reg
    sections_counts_by_class_copy[
        "Ratio Pred. from Lin. Reg. Using 3 Rcnt. Yrs."] = prediction_list_using_three_years_and_lin_reg
    sections_counts_by_class_copy["Ratio Pred. from Avg. Using All Data"] = prediction_list_using_all_years_and_average
    sections_counts_by_class_copy[
        "Ratio Pred. from Avg. Using 3 Rcnt. Yrs."] = prediction_list_using_3_years_and_average
    sections_counts_by_class_copy[
        "Ratio Pred. from Avg. Using Most Rcnt. Yr."] = prediction_list_using_1_year_and_average

    sections_counts_by_class_copy.to_excel("Course Statistics With Averages.xlsx")

def lin_reg_for_enrollment_ratio(model_course, all_class_data_df, how_far_we_are_looking):
    """
    create a linear regression model to obtain course enrollment predictions using various amounts of data

    :param model_course: course we are doing linear regression for
    :param all_class_data_df: df containing all of the classes' enrollment data
    :param how_far_we_are_looking: how much of the data we are using. "all" means using all avaialable years to make a prediction on the most recent year,
                                   while "3" means using only the three preceding years to make a prediction on the most recent year
    :return: the enrollment ratio prediction and the index of where it should go in the prediction array
    """
    course_we_are_running_model_on_df = all_class_data_df[(all_class_data_df["CRS Subject"] ==
                                                                  model_course[
                                                                      0]) &
                                                                 (all_class_data_df["CRS Course Number"] ==
                                                                  model_course[1]) &
                                                                 ((all_class_data_df["Academic Term"] ==
                                                                   model_course[2]))]

    course_we_are_running_model_on_df.drop(columns=["CRS Section Number", "Number Enrolled"], inplace=True)

    if len(course_we_are_running_model_on_df) < 4: # if there is less than 3 years of data, do not make a prediction-- just return -1
        return [-1], [course_we_are_running_model_on_df.index[len(course_we_are_running_model_on_df) - 1]]

    year_and_ratio_df = course_we_are_running_model_on_df[
        ["Academic Year", "Enrollment Ratio"]]  # extract only the year and ratio from the dataframe

    if model_course[2] == "Fall":
        date_objects = [datetime.date(int(x[:4]), 8, 1) for x in
                        year_and_ratio_df["Academic Year"]]  # converting fall years into datetime objects in order to make regression

    else:
        date_objects = [datetime.date(int(x[:4]), 1, 1) for x in
                        year_and_ratio_df["Academic Year"]]  # converting spring years into datetime objects

    year_and_ratio_df["Academic Year"] = date_objects

    year_and_ratio_df["Years From Start"] = np.arange(
        len(year_and_ratio_df.index))  # creating time-step column to perform linear regression

    X = year_and_ratio_df.loc[:,
        ['Years From Start']]  # creating features matrix, leaving out last row for training purposes
    X.drop(X.tail(1).index, inplace=True)
    y = year_and_ratio_df.loc[:, 'Enrollment Ratio']  # creating target vector, also leaving out last row
    y.drop(y.tail(1).index, inplace=True)

    if how_far_we_are_looking == "3":
        if len(X) > 3:  # for using only most recent 3 years to predict the actual recent year
            y.drop(y.head(len(y) - 3).index, inplace=True)
            X.drop(X.head(len(X) - 3).index, inplace=True)
            X["Years From Start"] = 0, 1, 2

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    print(r2_score(y_true=y, y_pred=model.predict(X)))

    # display = PredictionErrorDisplay(y_true=y, y_pred=model.predict(X))
    # display.plot()
    # plt.show()

    # print(course_we_are_running_model_on_df)
    # print(X)
    # print(y)
    # print(model.predict( (X.iloc[-1]["Years From Start"] + 1).reshape(-1,1)))
    # print(model.coef_, model.intercept_)

    graph_lin_reg(model, year_and_ratio_df, model_course, X)

    return model.predict((X.iloc[-1]["Years From Start"] + 1).reshape(-1, 1)), [year_and_ratio_df.iloc[-1].name] # return prediction and index of where to put prediction in prediction array used for excel file


def average_for_enrollment_ratio(model_course, all_class_data_df, how_far_we_are_looking):
    """
    other technique for predicting enrollment ratios.

    :param model_course: course we are predicting ratio for
    :param all_class_data_df: df containing all of the classes' enrollment data
    :param how_far_we_are_looking: "all" for using all available data; "3" for using only the three preceding years; "1" for using only the preceding year
    :return: predicted ratio and index for prediction array
    """
    course_we_are_averaging_df = all_class_data_df[(all_class_data_df["CRS Subject"] == model_course[ # get dataframe of course and term we are running model for
        0]) &
                                                          (all_class_data_df["CRS Course Number"] ==
                                                           model_course[1]) &
                                                          ((all_class_data_df["Academic Term"] ==
                                                            model_course[2]))]

    course_we_are_averaging_df.drop(columns=["CRS Section Number", "Number Enrolled"], inplace=True)

    if len(course_we_are_averaging_df) < 4: # if there's less than 4 years of data, do not make a prediction-- just return -1
        return [-1], [course_we_are_averaging_df.index[len(course_we_are_averaging_df) - 1]]

    year_and_ratio_df = course_we_are_averaging_df[
        ["Academic Year", "Enrollment Ratio"]]  # extract only the year and ratio from the dataframe

    index_to_return = year_and_ratio_df.iloc[-1].name

    year_and_ratio_df.drop(year_and_ratio_df.tail(1).index, inplace=True) # remove last year's ratio for training purposes

    if how_far_we_are_looking == "1":
        return [year_and_ratio_df["Enrollment Ratio"].iloc[-1]], [index_to_return]

    if how_far_we_are_looking == "3":
        if len(year_and_ratio_df) > 3:  # for using only most recent 3 years to predict the actual recent year
            year_and_ratio_df.drop(year_and_ratio_df.head(len(year_and_ratio_df) - 3).index, inplace=True)

    return [year_and_ratio_df['Enrollment Ratio'].sum() / len(year_and_ratio_df["Enrollment Ratio"])], [index_to_return]


def graph_lin_reg(model, year_and_ratio_df, model_course, X):

    year_and_ratio_df["Academic Year"] = year_and_ratio_df["Academic Year"].map(
        datetime.datetime.toordinal)  # for graphing

    year_and_ratio_df = year_and_ratio_df.drop(['Years From Start'], axis=1)  # remove Years From Start column to graph

    plt.figure(figsize=(16, 6))
    graph = sns.regplot(x=year_and_ratio_df["Academic Year"], y=year_and_ratio_df["Enrollment Ratio"],
                        data=year_and_ratio_df)

    graph.axes.set_title(f"{model_course[0]} {model_course[1]} {model_course[2]} Enrollment Ratios", fontsize=30)
    graph.set_xlabel('Date', fontsize=25)
    graph.set_ylabel('Enrollment Ratio', fontsize=22)
    graph.tick_params(labelsize=18)
    new_labels = [datetime.date.fromordinal(int(item)) for item in graph.get_xticks()]
    graph.set_xticklabels(new_labels)
    plt.gcf().subplots_adjust(bottom=0.13)

    plt.show()


def plot_a_class_section_frequency(classes_data, CRS_Subject_of_class, CRS_Course_Number_of_class):
    class_df = classes_data.loc[(classes_data["CRS Subject"] == CRS_Subject_of_class) & (
            classes_data["CRS Course Number"] == CRS_Course_Number_of_class)]

    sections_counts_by_class = class_df.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                 "Academic Term"])[["CRS Section Number"]].nunique()

    graph = sns.catplot(sections_counts_by_class, x="Academic Year", y=sections_counts_by_class.values.flatten(),
                        aspect=4.0, kind="bar",
                        hue='Academic Term')

    graph.fig.subplots_adjust(top=.94)

    graph.set(title=CRS_Subject_of_class + " " + CRS_Course_Number_of_class + " Section Frequency by Year and Term")
    plt.show()


def plot_a_class_enrollment(classes_data, CRS_Subject_of_class, CRS_Course_Number_of_class):
    class_df = classes_data.loc[(classes_data["CRS Subject"] == CRS_Subject_of_class) & (
            classes_data["CRS Course Number"] == CRS_Course_Number_of_class)]

    enrollment_counts_by_class = class_df.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                   "Academic Term"])[["CRS Section Number"]].count()

    graph = sns.catplot(enrollment_counts_by_class, x="Academic Year", y=enrollment_counts_by_class.values.flatten(),
                        aspect=4.0, kind="bar",
                        hue='Academic Term')

    graph.fig.subplots_adjust(top=.94)

    graph.set(title=CRS_Subject_of_class + " " + CRS_Course_Number_of_class + " Enrollment by Year and Term")
    plt.show()

def preprocess_student_data(student_data):
    student_data.drop(columns=["SFRSTCR_TERM_CODE", "CRS Subject", "CRS CRN", "CRS Course Number", "CRS Course Level",
                 "CRS Section Number",
                 "CRS Campus", "CRS Course Title", "CRS Primary Instructor PIDM", "CRS Grade", "CRS Mid Term Grade",
                 "CRS Schedule Desc", "REG Registered Hours", "REG Registration Status Code", "SGBSTDN_MAJR_CODE_1",
                 "MEET Building", "MEET Room Number", "MEET Begin Time", "MEET End Time", "MEET Meeting Days",
                  "GS Age at Enrollment", "GS Residency", "GS Student Type",
                               "First Generation Indicator", "Pell Eligible", "Sex", "GS Multiple Major Ind"],
        inplace=True)

    student_data = student_data.drop_duplicates(subset=['SPRIDEN_PIDM'])
    student_data.dropna(inplace=True)
    ord_enc = OrdinalEncoder()
    for columns in student_data.columns[3:]:
        student_data[columns] = ord_enc.fit_transform(student_data[[columns]])

    student_data.reset_index(inplace=True, drop=True)

    return student_data

def create_features_matrix_and_target_vector_for_rf_model(student_data, training_course):
    indices_of_students_in_training_course = []

    is_in_course = np.zeros(shape=(len(student_data), 1))

    for i in range(0, len(student_data)):
        if str(student_data.iloc[i]["SPRIDEN_PIDM"]) in training_course["SPRIDEN_PIDM"].to_string():
            indices_of_students_in_training_course.append(i)
            is_in_course[i] = 1

    student_data["Enrolled in Course Next Year"] = is_in_course

    students_in_training_course = []
    for i in indices_of_students_in_training_course:
        students_in_training_course.append(student_data.iloc[i])

    target_vector = np.ones(shape=(len(students_in_training_course), 1))

    features_matrix = pd.DataFrame(students_in_training_course)
    features_matrix["Enrolled in Course Next Year"] = target_vector

    features_matrix = features_matrix._append(student_data.sample(n=len(features_matrix))) # make so only non-students are sampled?

    return features_matrix.drop(columns="Enrolled in Course Next Year"), np.array(features_matrix["Enrolled in Course Next Year"])

def create_rf_model_for_course(all_student_data, course_subject, course_number):
    fall_2020_students_df = all_student_data.loc[
        (all_student_data["Academic Term"] == "Fall") & (all_student_data["Academic Year"] == "2020-2021")]

    fall_2020_students_df = preprocess_student_data(fall_2020_students_df)

    training_course = all_student_data.loc[
        (all_student_data["Academic Year"] == "2021-2022") & (all_student_data["Academic Term"] == "Fall")
        & (all_student_data["CRS Subject"] == course_subject) & (all_student_data["CRS Course Number"] == course_number)]

    fall_2020_students_df, target_vector = create_features_matrix_and_target_vector_for_rf_model(fall_2020_students_df, training_course)

    rf_model = RandomForestClassifier()
    rf_model.fit(fall_2020_students_df.drop(columns=["Academic Year", "Academic Term", "SPRIDEN_PIDM"]), target_vector)

    fall_2021_students_df = all_student_data.loc[
        (all_student_data["Academic Term"] == "Fall") & (all_student_data["Academic Year"] == "2021-2022")]

    fall_2021_students_df = preprocess_student_data(fall_2021_students_df)

    training_course = all_student_data.loc[
        (all_student_data["Academic Year"] == "2022-2023") & (all_student_data["Academic Term"] == "Fall")
        & (all_student_data["CRS Subject"] == course_subject) & (all_student_data["CRS Course Number"] == course_number)]

    target_vector = np.zeros(shape=(len(fall_2021_students_df), 1))

    for i in range(0, len(fall_2021_students_df)):
        if str(fall_2021_students_df.iloc[i]["SPRIDEN_PIDM"]) in training_course["SPRIDEN_PIDM"].to_string():
            target_vector[i] = 1

    y_pred = rf_model.predict(fall_2021_students_df.drop(columns=["Academic Year", "Academic Term", "SPRIDEN_PIDM"]))

    cm = confusion_matrix(target_vector, y_pred)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)

    correct_predictions = 0

    print(f"\n\nstudents in {course_subject} {course_number}")

    for i in range(0, len(target_vector)):
        if target_vector[i] == 1:
            print(fall_2021_students_df.iloc[i].to_frame().T.to_string()) # student in course
        correct_predictions += target_vector[i] and y_pred[i] # correct predictions

    print(f"\nstudents predicted to be in {course_subject} {course_number}")

    for i in range(0, len(target_vector)):
        if y_pred[i] == 1:
            print(fall_2021_students_df.iloc[i].to_frame().T.to_string())

    print(f"{int(sum(target_vector))} students from 2021 took the course in 2022. {int(correct_predictions)} predictions were correct. there were {fp} false positives and {fn} false negatives.")

def main():
    # classes_data = pd.read_excel("Course Data Set 6-26.xlsx")
    # classes_data.to_pickle("Course Data Set 6-26.pickle", compression="xz")
    #classes_data = pd.read_pickle("Course Data Set 6-26.pickle", compression="xz")

    # old_classes_data = pd.read_excel("Course Dataset for Summer Program June20.xlsx")
    # old_classes_data.to_pickle("class_data.pickle", compression="xz")
    #old_classes_data = pd.read_pickle("class_data.pickle", compression="xz")

    #classes_data = classes_data[classes_data["Academic Year"] != "2024-2025"]
    #export_course_statistics_to_xlsx(classes_data, old_classes_data)

    all_student_data = pd.read_pickle("class_data.pickle", compression="xz")

    # create_rf_model_for_course(all_student_data, "ENGL", "1101")
    # create_rf_model_for_course(all_student_data, "MATH", "2501") # calc 1
    # create_rf_model_for_course(all_student_data, "MATH", "3520") # linear
    # create_rf_model_for_course(all_student_data, "MATH", "2563") # transitions
    # create_rf_model_for_course(all_student_data, "COMP", "2270") # data structures
    # create_rf_model_for_course(all_student_data, "BSBA", "2209")


if __name__ == "__main__":
    main()
