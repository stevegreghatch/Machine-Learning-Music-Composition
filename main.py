import mapping
import trainingLibrary
import model
import dashboard

# set .mid and .midi to windows media player

# ---------------------------------------------------

# trainingLibrary.playRandomSelectionFromCorpus()

# trainingLibrary.playRandomSelectionFromPersonalCollection()

# ---------------------------------------------------

trainingLibrary.getDataFromAllSongsInFolder()
trainingLibrary.getDataFromRandomSelectionOfSongsFromCorpus()
mapping.mappingNoteToIntAllSongs()
model.LSTMFunction()
dashboard.dash()

# ---------------------------------------------------
