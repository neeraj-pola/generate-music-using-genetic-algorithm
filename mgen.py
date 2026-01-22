import click
from datetime import datetime
import os
import random
import time
from typing import List, Dict, Callable, Tuple

# Import music21 instead of pyo
from music21 import stream, note, chord, scale, midi, instrument, tempo

# Keep the genetic algorithm module
from algorithms.genetic import generate_genome, Genome, selection_pair, single_point_crossover, mutation

BITS_PER_NOTE = 4
KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SCALES = ["major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "majorBlues", "minorBlues"]


def int_from_bits(bits: List[int]) -> int:
    """Convert a list of bits to an integer."""
    return int(sum([bit*pow(2, index) for index, bit in enumerate(bits)]))


def genome_to_melody(genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                     pauses: bool, key: str, scale_name: str, root: int) -> Dict[str, list]:
    """Convert a genome to a melody representation."""
    # Split the genome into chunks representing individual notes
    notes = [genome[i * BITS_PER_NOTE:i * BITS_PER_NOTE + BITS_PER_NOTE] for i in range(num_bars * num_notes)]
    
    # Calculate note duration based on number of notes per bar
    note_length = 4 / float(num_notes)  # in quarter note lengths
    
    # Create a scale object from music21 based on the key string
    # First convert key name to pitch object
    key_pitch = note.Note(key).pitch
    
    # Create appropriate scale based on scale name
    if scale_name == "major":
        sc = scale.MajorScale(key_pitch)
    elif scale_name == "minor":
        sc = scale.MinorScale(key_pitch)
    elif scale_name == "dorian":
        sc = scale.DorianScale(key_pitch)
    elif scale_name == "phrygian":
        sc = scale.PhrygianScale(key_pitch)
    elif scale_name == "lydian":
        sc = scale.LydianScale(key_pitch)
    elif scale_name == "mixolydian":
        sc = scale.MixolydianScale(key_pitch)
    elif scale_name == "majorBlues":
        # Major blues scale - approximate with a concrete scale
        sc = scale.ConcreteScale(tonic=key_pitch, intervals=['P1', 'M2', 'm3', 'M3', 'P5', 'M6'])
    elif scale_name == "minorBlues":
        # Minor blues scale
        sc = scale.ConcreteScale(tonic=key_pitch, intervals=['P1', 'm3', 'P4', 'd5', 'P5', 'm7'])
    else:
        # Fallback to major
        sc = scale.MajorScale(key_pitch)
    
    # Make sure the scale has pitches
    if not sc.getPitches():
        # If no pitches, create a backup major scale
        sc = scale.MajorScale(key_pitch)
    
    # Initialize melody structure
    melody = {
        "notes": [],
        "velocity": [],
        "beat": []
    }
    
    # Convert each note from genome to musical parameters
    for note_bits in notes:
        integer = int_from_bits(note_bits)
        
        # Handle pauses flag
        if not pauses:
            integer = int(integer % pow(2, BITS_PER_NOTE - 1))
        
        # Check if this is a rest (highest bit set)
        if integer >= pow(2, BITS_PER_NOTE - 1):
            melody["notes"].append(0)  # 0 indicates rest
            melody["velocity"].append(0)
            melody["beat"].append(note_length)
        else:
            # Handle note extension (same note as previous)
            if len(melody["notes"]) > 0 and melody["notes"][-1] == integer:
                melody["beat"][-1] += note_length
            else:
                melody["notes"].append(integer)
                melody["velocity"].append(127)  # max velocity
                melody["beat"].append(note_length)
    
    # Generate different melodic steps/variations
    steps = []
    for step in range(num_steps):
        # Map integers to actual pitches in the scale
        step_notes = []
        for note_val in melody["notes"]:
            if note_val == 0:  # Rest
                step_notes.append(0)
            else:
                # Calculate pitch in scale, offset by step
                # Get the scale pitches - handle empty scale case
                scale_pitches = sc.getPitches()
                if not scale_pitches:
                    # Fallback if scale has no pitches
                    scale_pitches = scale.MajorScale(note.Note(key).pitch).getPitches()
                
                # Calculate scale index with safety check
                if len(scale_pitches) > 0:
                    scale_index = (note_val + step*2) % len(scale_pitches)
                    # Get the actual pitch
                    step_notes.append(scale_pitches[scale_index].midi)
                else:
                    # Complete fallback if still no pitches
                    step_notes.append(60 + note_val)  # Default to C scale starting at middle C
        steps.append(step_notes)
    
    melody["notes"] = steps
    return melody


def genome_to_stream(genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                      pauses: bool, key: str, scale_name: str, root: int, bpm: int) -> List[stream.Stream]:
    """Convert genome to music21 stream objects."""
    melody = genome_to_melody(genome, num_bars, num_notes, num_steps, pauses, key, scale_name, root)
    
    # Create a stream for each step/variation
    streams = []
    
    for step_notes in melody["notes"]:
        # Create a new stream
        s = stream.Stream()
        
        # Add tempo marking
        s.append(tempo.MetronomeMark(number=bpm))
        
        # Add instrument (default to piano)
        s.append(instrument.Piano())
        
        # Add notes to the stream
        current_offset = 0.0
        
        for i, midi_value in enumerate(step_notes):
            if midi_value == 0:  # Rest
                n = note.Rest()
                n.quarterLength = melody["beat"][i]
            else:
                n = note.Note(midi_value)
                n.volume.velocity = melody["velocity"][i]
                n.quarterLength = melody["beat"][i]
            
            # Add the note at the current offset
            s.insert(current_offset, n)
            current_offset += melody["beat"][i]
        
        streams.append(s)
    
    return streams


def fitness(genome: Genome, num_bars: int, num_notes: int, num_steps: int,
            pauses: bool, key: str, scale_name: str, root: int, bpm: int) -> int:
    """Play the melody and get user rating."""
    streams = genome_to_stream(genome, num_bars, num_notes, num_steps, pauses, key, scale_name, root, bpm)
    
    # Play the first stream (could modify to play more)
    try:
        # This will open the default MIDI player on the system
        streams[0].show('midi')
        
        # Wait for playback and user input
        rating = input("Rating (0-5): ")
        
        try:
            rating = int(rating)
        except ValueError:
            rating = 0
            
        return rating
    
    except Exception as e:
        print(f"Error playing MIDI: {e}")
        return 0


def save_genome_to_midi(filename: str, genome: Genome, num_bars: int, num_notes: int, num_steps: int,
                        pauses: bool, key: str, scale_name: str, root: int, bpm: int):
    """Save a genome as a MIDI file."""
    streams = genome_to_stream(genome, num_bars, num_notes, num_steps, pauses, key, scale_name, root, bpm)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save all streams as separate MIDI files
    for i, s in enumerate(streams):
        # Modify filename to include step number if there are multiple steps
        if len(streams) > 1:
            step_filename = f"{os.path.splitext(filename)[0]}_step{i}{os.path.splitext(filename)[1]}"
        else:
            step_filename = filename
            
        # Write to MIDI file
        s.write('midi', fp=step_filename)


@click.command()
@click.option("--num-bars", default=8, prompt='Number of bars:', type=int)
@click.option("--num-notes", default=4, prompt='Notes per bar:', type=int)
@click.option("--num-steps", default=1, prompt='Number of steps:', type=int)
@click.option("--pauses", default=True, prompt='Introduce Pauses?', type=bool)
@click.option("--key", default="C", prompt='Key:', type=click.Choice(KEYS, case_sensitive=False))
@click.option("--scale", default="major", prompt='Scale:', type=click.Choice(SCALES, case_sensitive=False))
@click.option("--root", default=4, prompt='Scale Root:', type=int)
@click.option("--population-size", default=10, prompt='Population size:', type=int)
@click.option("--num-mutations", default=2, prompt='Number of mutations:', type=int)
@click.option("--mutation-probability", default=0.5, prompt='Mutations probability:', type=float)
@click.option("--bpm", default=128, type=int)
def main(num_bars: int, num_notes: int, num_steps: int, pauses: bool, key: str, scale: str, root: int,
         population_size: int, num_mutations: int, mutation_probability: float, bpm: int):
    """Main function to run the genetic algorithm for music generation."""
    # Create folder for outputs named with current timestamp
    folder = str(int(datetime.now().timestamp()))
    
    # Generate initial population
    population = [generate_genome(num_bars * num_notes * BITS_PER_NOTE) for _ in range(population_size)]
    
    population_id = 0
    
    running = True
    while running:
        # Shuffle the population
        random.shuffle(population)
        
        # Evaluate fitness for each genome
        population_fitness = [(genome, fitness(genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)) 
                              for genome in population]
        
        # Sort by fitness
        sorted_population_fitness = sorted(population_fitness, key=lambda e: e[1], reverse=True)
        
        # Extract sorted population
        population = [e[0] for e in sorted_population_fitness]
        
        # Select top performers for next generation
        next_generation = population[0:2]
        
        # Generate offspring from the population
        for j in range(int(len(population) / 2) - 1):
            
            def fitness_lookup(genome):
                for e in population_fitness:
                    if e[0] == genome:
                        return e[1]
                return 0
            
            # Select parents and create offspring
            parents = selection_pair(population, fitness_lookup)
            offspring_a, offspring_b = single_point_crossover(parents[0], parents[1])
            offspring_a = mutation(offspring_a, num=num_mutations, probability=mutation_probability)
            offspring_b = mutation(offspring_b, num=num_mutations, probability=mutation_probability)
            next_generation += [offspring_a, offspring_b]
        
        print(f"Population {population_id} done")
        
        # Play the best genome
        print("Playing the #1 ranked melody...")
        streams = genome_to_stream(population[0], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        streams[0].show('midi')
        input("Press Enter after listening to continue...")
        
        # Play the second best
        print("Playing the #2 ranked melody...")
        streams = genome_to_stream(population[1], num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        streams[0].show('midi')
        input("Press Enter after listening to continue...")
        
        # Save MIDI files
        print("Saving population MIDI files...")
        os.makedirs(f"{folder}/{population_id}", exist_ok=True)
        for i, genome in enumerate(population):
            save_genome_to_midi(f"{folder}/{population_id}/{scale}-{key}-{i}.mid", 
                               genome, num_bars, num_notes, num_steps, pauses, key, scale, root, bpm)
        print("Done")
        
        # Check if we should continue
        running = input("Continue? [Y/n] ") != "n"
        population = next_generation
        population_id += 1


if __name__ == '__main__':
    main()