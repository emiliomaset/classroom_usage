import datetime
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

pd.options.mode.chained_assignment = None

def describe_data():
    print("""Academic Term: Spring, Fall, or Summer
Academic Year:
SFRSTCR_TERM_CODE: ?
CRS Subject: ?
CRS Subject Desc:
CRS Course Number:
CRS Section Number:
CRS CRN:
CRS Campus:
CRS CIP Code:
CRS Course Level:
CRS Course Title:
CRS Schedule Desc:
CRS Primary Instructor:
PIDM:
REG Part of Term:
GS Class Level:
GS Student ID:
GS Major Code:
GS Major:
GS Residency:
GS Student Type:
SFRSTCR_ADD_DATE:
MEET Building:
MEET Room Number:
MEET Meeting Days:
MEET Begin Time:
MEET End Time
""")


def get_graph_of_fall_enrollment_through_years(classes_data):  # all good
    """
    finding how many students were enrolled each fall semester through the years
    """
    fall_enrollment_through_the_years = dict((year, 0) for year in classes_data["Academic Year"].unique())

    for i, item in enumerate(fall_enrollment_through_the_years):  # assign fall enrollment to years
        # enrollment identified by count of unique student IDs
        fall_enrollment_through_the_years[item] = len(
            classes_data.groupby(["Academic Year", "Academic Term"])["GS Student ID"].unique().iloc[i * 3])

    plt.barh(fall_enrollment_through_the_years.keys(), fall_enrollment_through_the_years.values())
    plt.title("Academic years and their fall enrollments at FSU")
    plt.xlabel("Enrollment")
    plt.ylabel("Academic years")
    plt.gcf().subplots_adjust(left=0.18)
    plt.show()

    def export_classes_enrollment_and_sections_to_xlsx(claasses_data):
        test = classes_data.groupby(["CRS Subject", "CRS Course Number", "Academic Year", "Academic Term"])[
            ["CRS Section Number", "GS Student ID"]].nunique()
        test.to_excel("test.xlsx")


def export_course_statistics_to_xlsx(classes_data, old_classes_data):

    sections_counts_by_class = classes_data.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                     "Academic Term"])[["CRS Section Number"]].nunique()

    enrollment_counts_by_class = classes_data.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                       "Academic Term"])[["Enrollment"]].sum()

    sections_counts_by_class["Number Enrolled"] = enrollment_counts_by_class  # adding enrollment by course

    enrollment_by_year_and_term = old_classes_data.groupby(["Academic Year", "Academic Term"])[
        ["SPRIDEN_PIDM"]].nunique()  # counting enrollment by year and term

    number_enrolled_div_by_total_term_enrollment_ratio = np.zeros(shape=(13826, 1)) # create array to store ratio

    for i, index in enumerate(sections_counts_by_class.index):  # divide course enrollment by total enrollment for that year and term.
                                                                # I suppose this can be done in a more efficient way, but I'm not sure how to do it.
        number_enrolled_div_by_total_term_enrollment_ratio[i] = (sections_counts_by_class["Number Enrolled"].iloc[i] /
                                                                 int(enrollment_by_year_and_term.loc[(
                                                                 sections_counts_by_class.iloc[i].name[2],
                                                                 sections_counts_by_class.iloc[i].name[3])]))


    sections_counts_by_class["Enrollment Ratio"] = number_enrolled_div_by_total_term_enrollment_ratio

    sections_counts_by_class_copy = sections_counts_by_class.copy() # make copy to use when exporting to excel
                                                                    # makes excel file easier to read

    sections_counts_by_class.reset_index(inplace=True) # reset index to use indices for filling array of predictions

    prediction_list = np.zeros(shape=(13826, 1))

    for i, item in enumerate(sections_counts_by_class.groupby(["CRS Subject", "CRS Course Number", "Academic Term"])[
                                 "Enrollment Ratio"].unique().index):

        predictions, indices_for_sheet = lin_reg_for_enrollment_ratio(item, sections_counts_by_class) # run a model for each course

        for i, index in enumerate(indices_for_sheet): # populate predictions array
            prediction_list[index] = predictions[i]


    sections_counts_by_class_copy["Ratio Prediction from Lin. Reg"] = prediction_list

    sections_counts_by_class_copy.to_excel("Course Statistics.xlsx")


def lin_reg_for_enrollment_ratio(model_course, sections_counts_by_class):

    course_we_are_running_model_on_df = sections_counts_by_class[ (sections_counts_by_class["CRS Subject"] == model_course[0]) & # get dataframe of course and term we are running model for
        (sections_counts_by_class["CRS Course Number"] == model_course[1]) &
        ((sections_counts_by_class["Academic Term"] == model_course[2]))]

    course_we_are_running_model_on_df.drop(columns=["CRS Section Number", "Number Enrolled"], inplace=True)

    if len(course_we_are_running_model_on_df) < 4:
        return [-1], [course_we_are_running_model_on_df.index[len(course_we_are_running_model_on_df) - 1]]

    year_and_ratio_df = course_we_are_running_model_on_df[["Academic Year", "Enrollment Ratio"]]  # extract only the year and ratio from the dataframe

    if model_course[2] == "Fall":
        date_objects = [datetime.date(int(x[:4]), 8, 1) for x in
                        year_and_ratio_df["Academic Year"]]  # converting fall years into datetime objects

    else:
        date_objects = [datetime.date(int(x[:4]), 1, 1) for x in
                        year_and_ratio_df["Academic Year"]]  # converting spring years into datetime objects

    year_and_ratio_df["Academic Year"] = date_objects

    year_and_ratio_df["Years From Start"] = np.arange(len(year_and_ratio_df.index))  # creating time-step column to perform linear regression

    X = year_and_ratio_df.loc[:, ['Years From Start']]  # creating features matrix, leaving out last row for testing purposes
    X.drop(X.tail(1).index, inplace=True)
    y = year_and_ratio_df.loc[:, 'Enrollment Ratio'] # creating target vector
    y.drop(y.tail(1).index, inplace=True)

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    #graph_lin_reg(model, year_and_ratio_df, model_course, X)

    return model.predict(year_and_ratio_df.iloc[-1]["Years From Start"].reshape(-1,1)), [year_and_ratio_df.iloc[-1].name]
def graph_lin_reg(model, year_and_ratio_df, model_course, X):
    y_pred = pd.Series(model.predict(X), index=year_and_ratio_df["Academic Year"])

    #print(f'mean squared error: {mean_squared_error(y, y_pred):.15f}')


    year_and_ratio_df["Academic Year"]= year_and_ratio_df["Academic Year"].map(datetime.datetime.toordinal) # for graphing
    # y_pred.index = y_pred.index.map(datetime.datetime.toordinal) # for graphing?

    year_and_ratio_df.drop(['Years From Start'], axis=1, inplace=True) # remove Years From Start column to graph

    plt.figure(figsize=(12, 6))
    graph = sns.regplot(x=year_and_ratio_df["Academic Year"], y=year_and_ratio_df["Enrollment Ratio"], data=year_and_ratio_df)

    graph.set_xlabel('Date')
    graph.set_ylabel('Enrollment Ratio')
    new_labels = [datetime.date.fromordinal(int(item)) for item in graph.get_xticks()]
    graph.set_xticklabels(new_labels)

    graph.set(title= f"{model_course[0]} { model_course[1]} {model_course[2]} Enrollment Ratios")

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


def main():
    #classes_data = pd.read_excel("Course Data Set 6-26.xlsx")
    #classes_data.to_pickle("Course Data Set 6-26.pickle", compression="xz")

    classes_data = pd.read_pickle("Course Data Set 6-26.pickle", compression="xz")
    old_classes_data = pd.read_pickle("class_data.pickle", compression="xz")
    classes_data = classes_data[classes_data["Academic Year"] != "2024-2025"]

    export_course_statistics_to_xlsx(classes_data, old_classes_data)

    #plot_a_class_enrollment(classes_data, "ENGL", "1101")
    #plot_a_class_section_frequency(classes_data, "ENGL", "1101")


if __name__ == "__main__":
    main()
