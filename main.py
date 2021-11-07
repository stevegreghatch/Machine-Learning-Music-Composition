import mapping
import modelV3
import trainingLibrary

# set .mid and .midi to windows media player

# ---------------------------------------------------

# trainingLibrary.playRandomSelectionFromCorpus()

# trainingLibrary.playRandomSelectionFromPersonalCollection()

# ---------------------------------------------------

# trainingLibrary.getDataFromRandomSelectionOfSongsFromCorpus()

trainingLibrary.getDataFromAllSongsInFolder()

# ---------------------------------------------------

mapping.mappingNoteToIntAllSongs()

# ---------------------------------------------------

modelV3.LSTMFunction()
