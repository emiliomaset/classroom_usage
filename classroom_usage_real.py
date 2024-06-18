import numpy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
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
    part_of_term_distribution = classes_data.groupby(['Academic Year', 'Academic Term', 'REG Part of Term'])["GS Student ID"].nunique() # stopped working on this because there's no spring full term data?
    print(part_of_term_distribution.index)
    plt.barh(part_of_term_distribution.unstack().index, part_of_term_distribution)


    with sns.axes_style('white'):
        g = sns.catplot(x=part_of_term_distribution.unstack().index, y= part_of_term_distribution)

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

def export_all_classes_enrollemnts_and_sections_to_xlsx(classes_data): # all good

    enrollment_and_sections_counts_by_class = classes_data.groupby(["CRS Subject", "CRS Course Number", "Academic Year",
                                                                    "Academic Term"])[["CRS Section Number", "GS Student ID"]].nunique()
    enrollment_and_sections_counts_by_class.to_excel("Course Statistics.xlsx")

def main():
    classes_data = pd.read_csv("Course Dataset for Summer Program.csv")

    print(classes_data[classes_data["CRS Course Number"] == "1105"].to_string()) # only 12 rows?

    #find_parts_of_term_distribution(classes_data)
    #find_meeting_time_popularity(classes_data)
    export_all_classes_enrollemnts_and_sections_to_xlsx(classes_data)

if __name__ == "__main__":
    main()
