import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

def get_bar_graph_of_total_enrollment_by_total_year(classes_data): # all good
    """
    finding and creating a bar graph of total enrollment by year. that is, the sum of how many courses each student took
    """
    year_vs_enrollment = classes_data["Academic Year"].value_counts()
    year_vs_enrollment = year_vs_enrollment.sort_index()
    plt.barh(year_vs_enrollment.keys(), year_vs_enrollment.values)
    plt.title("Academic years and their enrollments (Fall, Spring, Summer) at FSU")
    plt.xlabel("Total enrollment among all courses")
    plt.ylabel("Academic years")
    plt.gcf().subplots_adjust(left=0.18)
    plt.show()

def find_meeting_time_popularity(classes_data):
    """
    finding the frequencies of meet times during the full term.
    grouping the data by academic year, then by part of term (wishing to do only full), and then counting the frequency of meet times
    """
    meeting_time_frequencies = classes_data.groupby(["Academic Year", "REG Part of Term"])[
        "MEET Begin Time"].value_counts()
    # how to do only full term? and then how to graph?
    print(meeting_time_frequencies)
    #print(meeting_time_frequencies.keys())
    #print(meeting_time_frequencies.values)

def find_parts_of_term_distribution(classes_data):
    """
    finding how many students participate in each term of the semester between
    the full term, reg 1st 7 or 8 Weeks, and reg 2nd 7 or 8 Weeks
    """

    test = classes_data.loc[ (classes_data["REG Part of Term"] == "Full Term" ) | (classes_data["REG Part of Term"] == "Reg 1st 7 or 8 Weeks") | (classes_data["REG Part of Term"] == "Reg 2nd 7 or 8 Weeks") ]
    # test.groupby(['Academic Year', 'Academic Term', "REG Part of Term"])["GS Student ID"].nunique().plot(kind='bar') # stopped working on this because there's no spring full term data?
    #
    # plt.show()
    #plt.barh(part_of_term_distribution.unstack().columns, 10)


    with sns.axes_style('white'):
        g = sns.catplot(test, x = "Academic Year", aspect=4.0, kind='count',
                           hue='REG Part of Term')

    plt.show()



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

def get_graph_of_fall_enrollment_through_years(classes_data): # all good
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

def export_course_statistics_to_xlsx(classes_data): # all good

    sections_counts_by_class = classes_data.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                                    "Academic Term"])[["CRS Section Number"]].nunique()

    enrollment_counts_by_class = classes_data.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                                    "Academic Term"])[["CRS Section Number"]].count()

    sections_counts_by_class["Number Enrolled"] = enrollment_counts_by_class # adding enrollment by course

    enrollment_by_year_and_term = classes_data.groupby(["Academic Year", "Academic Term"])[["SPRIDEN_PIDM"]].nunique() # counting enrollment by year and term

    number_enrolled_div_by_total_term_enrollment_ratio = np.zeros(shape=(15388, 1))

    for i, index in enumerate(sections_counts_by_class.index): # divide course enrollment by total enrollment for that year and term
       number_enrolled_div_by_total_term_enrollment_ratio[i] = (sections_counts_by_class["Number Enrolled"].iloc[i] /
                                                                int(enrollment_by_year_and_term.loc[(sections_counts_by_class.iloc[i].name[2],sections_counts_by_class.iloc[i].name[3])]))

    sections_counts_by_class["Ratio of Enrolled"] = number_enrolled_div_by_total_term_enrollment_ratio

    test = sections_counts_by_class.iloc[(sections_counts_by_class.index.get_level_values(0).str.contains('ENGL')) & (sections_counts_by_class.index.get_level_values(1).str.contains('1101')) &
                                         (sections_counts_by_class.index.get_level_values(3).str.contains('Fall'))]

    plt.figure(figsize=(12,6))

    graph = sns.scatterplot(test, x="Academic Year", y=test["Ratio of Enrolled"])

    graph.set(title="ENGL 1101 Fall Enrollment Ratios")

    plt.show()

    sections_counts_by_class.to_excel("Course Statistics.xlsx")

def plot_a_class_section_frequency(classes_data, CRS_Subject_of_class, CRS_Course_Number_of_class):
    class_df = classes_data.loc[(classes_data["CRS Subject"] == CRS_Subject_of_class) & (classes_data["CRS Course Number"] == CRS_Course_Number_of_class)]

    sections_counts_by_class = class_df.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                     "Academic Term"])[["CRS Section Number"]].nunique()

    graph = sns.catplot(sections_counts_by_class, x="Academic Year", y=sections_counts_by_class.values.flatten(), aspect=4.0, kind="bar",
                        hue='Academic Term')

    graph.fig.subplots_adjust(top=.94)

    graph.set(title=CRS_Subject_of_class + " " + CRS_Course_Number_of_class + " Section Frequency by Year and Term")
    plt.show()

def plot_a_class_enrollment(classes_data, CRS_Subject_of_class, CRS_Course_Number_of_class):
    class_df = classes_data.loc[(classes_data["CRS Subject"] == CRS_Subject_of_class) & (classes_data["CRS Course Number"] == CRS_Course_Number_of_class)]

    enrollment_counts_by_class = class_df.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                       "Academic Term"])[["CRS Section Number"]].count()

    graph = sns.catplot(enrollment_counts_by_class, x="Academic Year", y=enrollment_counts_by_class.values.flatten(), aspect=4.0, kind="bar",
                        hue='Academic Term')

    graph.fig.subplots_adjust(top=.94)

    graph.set(title=CRS_Subject_of_class + " " + CRS_Course_Number_of_class + " Enrollment by Year and Term")
    plt.show()

def main():
    classes_data = pd.read_pickle("class_data.pickle", compression="xz")

    export_course_statistics_to_xlsx(classes_data)

    #plot_a_class_enrollment(classes_data, "ENGL", "1101")
    #plot_a_class_section_frequency(classes_data, "ENGL", "1101")

if __name__ == "__main__":
    main()
