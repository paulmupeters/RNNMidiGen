import time
import glob
from keras import utils, Sequential, layers
from keras.callbacks import ModelCheckpoint
from mido import MidiFile,tick2second
import numpy as np

MAX_TIME_ALLOWED = 300
def getNotes(timeMido, veloMido, notesMido):
    start= time.time()
    print("getting midi files")
    folders = ["2004",  "2006"]#,  "2008",  "2009",  "2011",  "2013",  "2014",  "2015",  "2017"]
    files = 0
    for folder in folders:
        #path = "/home/paul/Projects/maestro-v1.0.0/"+ folder +"/*.midi"
        path = "/data/s1453440/maestro-v1.0.0-midi/maestro-v1.0.0/"+ folder +"/*.midi"
        for file in glob.glob(path):
            files +=1
            if (time.time() - start) > MAX_TIME_ALLOWED:
                break
            midi = MidiFile(file)
            for track in midi.tracks:
                for msg in track:
                    if msg.type == 'set_tempo':
                        tempo = msg.tempo
                    if not msg.is_meta and msg.type == 'note_on':
                        notesMido.append(msg.note)
                        timeMido.append(msg.time)
                        veloMido.append(msg.velocity)
    return notesMido, timeMido, veloMido

def notes_network_in_out(notesMido):
    print("creating input and output for notes")
    inputLength = 80
    # get all pitch names
    notes = np.unique(notesMido)
    # create a dictionary to map pitches to integers
    normNotes = np.array(notesMido) - 21
    network_input = []
    network_output = []
    # create input sequences and the corresponding outputs
    for i in range(0, len(normNotes) - inputLength):
        network_input.append(normNotes[i:i+inputLength])
        network_output.append(normNotes[i + inputLength])
    n_patterns = len(network_input)
    # reshape the input into a format compatible with LSTM layers
    network_input = np.reshape(network_input, (n_patterns ,inputLength, 1))
    # normalize input
    network_input = network_input / float(len(notes))
    network_output = utils.to_categorical(network_output)
    return network_input, network_output

def net_in_out_time(times,timeMido, normNotesMido, normTimeMido ,inputLength = 80):
    network_input = []
    network_output = []

    time_to_int = dict((time,integer) for integer, time in enumerate(times))
    for i in range(0, len(timeMido) - inputLength):
        network_input.append( [normNotesMido[i + 1:i + inputLength + 1], normTimeMido[i:i+inputLength]])
        network_output.append(time_to_int[timeMido[i + inputLength]])
    
    n_patterns = len(network_input)
    network_input = np.array(network_input)
    net_in = []
    for i in range(0, n_patterns):
        net_in.append(network_input[i].transpose())
    network_input = np.array(net_in)
    network_output = utils.to_categorical(network_output)
    
    return network_input, network_output
    


def main():
    timeMido = []
    veloMido = []
    notesMido = []
    notesMido, timeMido, _ = getNotes(timeMido, veloMido, notesMido)
    
    #times = np.unique(timeMido)
    notes = np.unique(notesMido)

    #time_to_int = dict((time, integer) for integer, time in enumerate(times))
    #normNotesMido = np.array(notesMido)/float(len(notes))
    #normTimeMido = np.array(timeMido)/float(len(times))

    network_input, network_output = notes_network_in_out(notesMido)
    
    
    model = Sequential()
    model.add(layers.LSTM(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(256, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(256))
    model.add(layers.Dense(256))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(len(notes)))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    filepath = "/data/s1453440/weights-improved-{epoch:02d}-{loss:.4f}-bigger.hdf5"    

    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]     

    model.fit(network_input, network_output, epochs=2, batch_size=64, callbacks=callbacks_list)
    
    
    model.save("/data/s1453440/myModel.h5")

    
    """
    
    model.load_weights('/home/s1453440/weights-improvement-10-3.3827-bigger.hdf5')
    
    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    # generate 500 notes
    for note_index in range(500):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]
    """
    





if __name__ == "__main__":
    main()
