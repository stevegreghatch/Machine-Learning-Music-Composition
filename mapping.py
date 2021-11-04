import numpy

import trainingLibrary

# map noteAndDuration elements to int value while maintaining order of composition

noteToIntWithData = []
noteToIntList = []
noteToIntLists = []

def mappingOneSong():
    # get extracted data
    listOfAllData = trainingLibrary.listOfNotesAndDurations
    # get unique number for each note:duration element
    # get unique elements
    uniqueValues = numpy.unique(listOfAllData)
    # give each unique element a number
    for numberAndData in (enumerate(uniqueValues)):
        noteToIntWithData.append(numberAndData)
    # match each element with its number
    for data in listOfAllData:
        for numberAndData in (enumerate(uniqueValues)):
            if data == numberAndData[1]:
                data = numberAndData[0]
                noteToIntList.append(data)
    print(noteToIntList)

    # ---------------------------------------------

def mappingNoteToIntAllSongs():
    # get extracted data
    # get list of all elements from all lists in listOfAllNotesAndDurations
    listOfAllElements = []
    for listOfNotesAndDurations in trainingLibrary.listOfAllNotesAndDurations:
        for element in listOfNotesAndDurations:
            listOfAllElements.append(element)
    # get unique number for each note:duration element
    # get unique elements
    uniqueValues = numpy.unique(listOfAllElements)
    # give each unique element a number
    for numberAndData in enumerate(uniqueValues):
        noteToIntWithData.append(numberAndData)
    # match each element with its number and convert back into list of lists
    for listOfNotesAndDurations in trainingLibrary.listOfAllNotesAndDurations:
        newSongList = []
        for element in listOfNotesAndDurations:
            for numberAndData in enumerate(uniqueValues):
                # print(numberAndData)
                if element == numberAndData[1]:
                    element = numberAndData[0]
                    noteToIntList.append(element)
                    newSongList.append(element)
        noteToIntLists.append(newSongList)
    # print(noteToIntWithData)
    # print('\n')
    # print(noteToIntList)
    # print('\n')
    # print(noteToIntLists)

    # ---------------------------------------------

# TRAIL NUMBER 2

