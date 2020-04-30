from mido import MidiFile,tick2second
import time
import glob
from keras import utils, Sequential, layers
from keras.callbacks import ModelCheckpoint
import numpy as np


def getNotes(timeMido, veloMido, notesMido):
    start= time.time()
    print("getting midi files")
    folders = ["2004"]#,  "2008",  "2009",  "2011",  "2013",  "2014",  "2015",  "2017"]
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


def main():
    timeMido = []
    veloMido = []
    notesMido = []
    
    notesMido, _, _ = getNotes(timeMido, veloMido, notesMido)
    network_input, network_output = notes_network_in_out(notesMido)
    notes = np.unique(notesMido)   
    model = Sequential()
    model.add(layers.LSTM(
        256,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(512, return_sequences=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(256))
    model.add(layers.Dense(256))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(len(notes)))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"    

    checkpoint = ModelCheckpoint(
        filepath, monitor='loss', 
        verbose=0,        
        save_best_only=True,        
        mode='min'
    )    
    callbacks_list = [checkpoint]     

    model.fit(network_input, network_output, epochs=10, batch_size=64, callbacks=callbacks_list)





if __name__ == "__main__":
    main()