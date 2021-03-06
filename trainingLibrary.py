import copy
import random
import os

from music21 import *

listOfNotesAndDurations = []
listOfAllNotesAndDurations = []

def getListOfNotesAndDurations(work):
    listOfNotesAndDurations.clear()
    noteToAdd = None
    notesToParse = work.flat.notes
    for element in notesToParse:
        if isinstance(element, note.Note):
            noteToAdd = str(element.pitch)
        elif isinstance(element, chord.Chord):
            noteToAdd = str([n.nameWithOctave for n in element.pitches])
        # time relative to quarter note = 1.0
        durationOfNote = str(element.quarterLength)
        listOfNotesAndDurations.append(noteToAdd + ':' + durationOfNote)
    # print(listOfNotesAndDurations)
    return listOfNotesAndDurations


# training set 1 -------------------------------------------------------------------------------------------------------
# corpus files from music21 - classical composers

listOfComposersInCorpus = ['bach', 'beach', 'beethoven', 'chopin', 'ciconia', 'corelli', 'cpebach',
                           'handel', 'haydn', 'joplin', 'josquin', 'monteverdi', 'mozart',
                           'palestrina', 'schoenberg', 'schubert', 'schumann', 'schumann_clara',
                           'verdi', 'weber']

# function for all data -----------------------------------------------
def getDataFromRandomSelectionOfSongsFromCorpus():
    listOfPathsOfFilesByComposer = []
    listOfPathsToChooseFrom = []
    listOfChosenPaths = []

    # get paths for all files by selected composers
    for composer in listOfComposersInCorpus:
        pathsOfFilesByComposer = corpus.getComposer(composer)
        listOfPathsOfFilesByComposer.append(pathsOfFilesByComposer)
    for filePaths in listOfPathsOfFilesByComposer:
        for filePath in filePaths:
            listOfPathsToChooseFrom.append(filePath)

    # selects desired number of random paths from selected composers from Corpus
    # 116 chosen to match number of midi files from personal collection
    for i in range(116):
        chosenPath = random.choice(listOfPathsToChooseFrom)
        if chosenPath not in listOfChosenPaths:
            listOfChosenPaths.append(chosenPath)

    # get data from each path
    for path in listOfChosenPaths:
        work = converter.parse(path)
        # print('\n')
        # work.show('text')
        title = os.path.basename(os.path.dirname(path)).capitalize() + ' - ' + os.path.basename(path)
        data = getListOfNotesAndDurations(work)
        print(title)
        # print(data)
        # print('\n')
        listOfAllNotesAndDurations.append(copy.deepcopy(data))
    # print(listOfAllNotesAndDurations)

# training set 2 -------------------------------------------------------------------------------------------------------
# personal collection
# function for all data -----------------------------------------------
def getDataFromAllSongsInFolder():
    listOfFilePaths = []
    folder = 'insert path here'  # update path
    listOfFiles = os.listdir(folder)

    # get list of file paths
    for file in listOfFiles:
        filePath = folder + file
        listOfFilePaths.append(filePath)

    # get data from each path
    for path in listOfFilePaths:
        work = converter.parse(path)
        title = os.path.basename(path)
        data = getListOfNotesAndDurations(work)
        print(title)
        # print(data)
        # print('\n')
        listOfAllNotesAndDurations.append(copy.deepcopy(data))
        # work.show('midi')
        # work.show('text')
    print(listOfAllNotesAndDurations)
