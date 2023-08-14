from datasets.preprocessing.openrooms_preprocessing import OpenroomPublicPreprocessing

processor = OpenroomPublicPreprocessing(modes=['validation'])

processor.process_file(processor.files['validation'][3], 'validation')