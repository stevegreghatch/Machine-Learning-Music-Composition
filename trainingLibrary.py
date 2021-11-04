import copy
import random
import os
import logging
import numpy
import mapping
import collections

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
    for i in range(400):
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

# functions for random selection -------------------------------
def returnRandomPathFromPersonalCollection():
    folder = '-Desktop/Midi_Files/'
    listOfFiles = os.listdir(folder)
    listOfFilePaths = []
    for file in listOfFiles:
        filePath = folder + file
        listOfFilePaths.append(filePath)
    selectedPath = random.choice(listOfFilePaths)
    return selectedPath

def playRandomSelectionFromPersonalCollection():
    for i in range(10):
        selectedPath = str(returnRandomPathFromPersonalCollection())
        work = converter.parse(selectedPath)
        title = os.path.basename(os.path.normpath(selectedPath))
        data = getListOfNotesAndDurations(work)
        print(title)
        # print(data)
        # print('\n')
        # work.show('midi')
        # work.show('text')
        listOfAllNotesAndDurations.append(copy.deepcopy(data))


# function for all data -----------------------------------------------
def getDataFromAllSongsInFolder():
    listOfFilePaths = []
    folder = '-Desktop/Midi_Files/'
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
    # print(listOfAllNotesAndDurations)
