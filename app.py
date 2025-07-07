import argparse
import logging
import yaml
from pathlib import Path
import tqdm
import time
from PIL import Image

from clams import ClamsApp, Restifier
from clams.appmetadata import AppMetadata
from mmif import Mmif, View, Document, AnnotationTypes, DocumentTypes
from mmif.utils import video_document_helper as vdh
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class MoondreamCaptioner(ClamsApp):

    def __init__(self):
        super().__init__()
        
        # Check if CUDA is available
        if torch.cuda.is_available():
            device = "cuda"
            self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            self.logger.info("CUDA not available, using CPU")
        
        self.device = device
        
        model_id = "vikhyatk/moondream2"
        revision = "2025-06-21"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision=revision,
            device_map={"": device}
        )
        
        # If using CPU, explicitly move model to CPU
        if device == "cpu":
            self.model = self.model.to("cpu")

    def _appmetadata(self) -> AppMetadata:
        pass
    
    def load_config(self, config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def get_prompt(self, label: str, parameters: dict) -> str:
        if 'promptMap' in parameters and parameters['promptMap']:
            for mapping in parameters['promptMap']:
                if ':' in mapping:
                    map_label, map_prompt = mapping.split(':', 1)
                    if map_label == label:
                        return map_prompt
        if 'defaultPrompt' in parameters:
            return parameters['defaultPrompt']
        return ""

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        self.logger.debug(f"Annotating with parameters: {parameters}")
        config_file = parameters.get('config')
        self.logger.debug(f"config_file: {config_file}")
        if config_file:
            config_dir = Path(__file__).parent
            config_file_path = config_dir / config_file
            config = self.load_config(config_file_path)
            if 'default_prompt' in config:
                parameters['defaultPrompt'] = config['default_prompt']
            if 'custom_prompts' in config:
                prompt_map = []
                for label, prompt in config['custom_prompts'].items():
                    prompt_map.append(f"{label}:{prompt}")
                parameters['promptMap'] = prompt_map
        else:
            config = {}
        if 'context_config' not in config:
            config['context_config'] = {
                'input_context': 'timeframe',
                'timeframe': {
                    'app_uri': 'http://apps.clams.ai/swt-detection/',
                    'label_mapping': {},
                    'ignore_other_labels': False
                }
            }
        batch_size = 16  # Moondream batch size
        new_view: View = mmif.new_view()
        self.sign_view(new_view, parameters)
        new_view.new_contain(DocumentTypes.TextDocument)
        new_view.new_contain(AnnotationTypes.Alignment)

        def process_batch(prompts_batch, images_batch, annotations_batch):
            try:
                # Process the batch of images and prompts using Moondream's caption method
                answers = []
                for image, prompt in zip(images_batch, prompts_batch):
                    # The prompt may be empty if not provided
                    if prompt:
                        result = self.model.caption(image, prompt)
                    else:
                        result = self.model.caption(image)
                    # result["caption"] may be a string or a generator (streaming)
                    if isinstance(result["caption"], str):
                        answers.append(result["caption"])
                    else:
                        # If streaming, join all tokens
                        answers.append(''.join(result["caption"]))
                for result, annotation in zip(answers, annotations_batch):
                    text_document = new_view.new_textdocument(result.strip())
                    alignment = new_view.new_annotation(AnnotationTypes.Alignment)
                    alignment.add_property("source", annotation['source'])
                    alignment.add_property("target", text_document.long_id)
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        input_context = config['context_config']['input_context']

        if input_context == "image":
            image_docs = mmif.get_documents_by_type(DocumentTypes.ImageDocument)
            for i in range(0, len(image_docs), batch_size):
                batch_docs = image_docs[i:i + batch_size]
                prompts = [self.get_prompt('default', parameters)] * len(batch_docs)
                images = [Image.open(doc.location_path()) for doc in batch_docs]
                annotations_batch = [{'source': doc.long_id} for doc in batch_docs]
                start_time = time.time()
                process_batch(prompts, images, annotations_batch)
                self.logger.debug(f"Processed batch of {len(batch_docs)} in {time.time() - start_time:.2f} seconds")

        elif input_context == 'timeframe':
            self.logger.debug(f"input_context: {input_context}")
            app_uri = config['context_config']['timeframe']['app_uri']
            all_views = mmif.get_all_views_contain(AnnotationTypes.TimeFrame)
            for view in all_views:
                self.logger.debug(f"view.metadata.app: {view.metadata.app}")
                if app_uri in view.metadata.app:
                    self.logger.debug(f"found view with app_uri: {app_uri}")
                    timeframes = view.get_annotations(AnnotationTypes.TimeFrame)
                    break
            label_mapping = config['context_config']['timeframe'].get('label_mapping', {})
            ignore_other_labels = config['context_config']['timeframe'].get('ignore_other_labels', False)

        elif input_context == 'fixed_window':
            self.logger.debug(f"input_context: {input_context}")
            video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
            window_duration = config['context_config']['fixed_window']['window_duration']
            stride = config['context_config']['fixed_window']['stride']
            try:
                fps = float(video_doc.get_property('fps'))
            except:
                fps = 29.97
            try:
                total_frames = int(video_doc.get_property('frameCount'))
            except:
                total_frames = int(29.97*60*60)
            frame_numbers = list(range(0, total_frames, int(fps * stride)))
        else:
            raise ValueError(f"Unsupported input context: {input_context}")

        if input_context == 'timeframe':
            timeframes = list(timeframes)
            if ignore_other_labels:
                timeframes = [tf for tf in timeframes if tf.get_property('label') in label_mapping]
                if not timeframes:
                    self.logger.warning("No timeframes found with labels matching the label_mapping")
                    return mmif
            for timeframe in timeframes:
                timeframe.add_property('timeUnit', 'milliseconds')
            all_frame_numbers = [vdh.get_representative_framenum(mmif, timeframe) for timeframe in timeframes]
            self.logger.debug(f"Extracted frame numbers: {all_frame_numbers}")
            video_doc = mmif.get_documents_by_type(DocumentTypes.VideoDocument)[0]
            if not video_doc:
                raise ValueError("No video document found in MMIF")
            try:
                temp_frame_numbers = all_frame_numbers.copy()
                all_images = vdh.extract_frames_as_images(video_doc, temp_frame_numbers, as_PIL=True)
                self.logger.debug(f"Successfully extracted {len(all_images)} images")
                if len(all_images) != len(all_frame_numbers):
                    self.logger.warning(f"Warning: Number of extracted images ({len(all_images)}) doesn't match number of frame numbers ({len(all_frame_numbers)})")
            except Exception as e:
                self.logger.error(f"Error extracting frames: {str(e)}")
                raise
            for i in tqdm.tqdm(range(0, len(timeframes), batch_size)):
                batch_timeframes = timeframes[i:i + batch_size]
                batch_images = all_images[i:i + batch_size]
                prompts = []
                annotations_batch = []
                for timeframe in batch_timeframes:
                    label = timeframe.get_property('label')
                    mapped_label = label_mapping.get(label, 'default')
                    prompt = self.get_prompt(mapped_label, parameters)
                    prompts.append(prompt)
                    representative_id = timeframe.get_property('representatives')[0]
                    annotations_batch.append({'source': representative_id})
                start_time = time.time()
                process_batch(prompts, batch_images, annotations_batch)
                self.logger.debug(f"Processed batch of {len(batch_timeframes)} in {time.time() - start_time:.2f} seconds")

        elif input_context == 'fixed_window':
            prompts = []
            images_batch = []
            annotations_batch = []
            for frame_number in tqdm.tqdm(frame_numbers):
                try:
                    image = vdh.extract_frames_as_images(video_doc, [frame_number], as_PIL=True)[0]
                except:
                    self.logger.warning(f"Failed to extract frame_number: {frame_number}")
                    continue
                prompt = self.get_prompt('default', parameters)
                prompts.append(prompt)
                images_batch.append(image)
                timepoint = new_view.new_annotation(AnnotationTypes.TimePoint)
                timepoint.add_property("timePoint", frame_number)
                annotations_batch.append({'source': timepoint.long_id})
                if len(prompts) == batch_size:
                    start_time = time.time()
                    process_batch(prompts, images_batch, annotations_batch)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    self.logger.debug(f"Processed a batch of {batch_size} in {elapsed_time:.2f} seconds.")
                    prompts, images_batch, annotations_batch = [], [], []
            if prompts:
                start_time = time.time()
                process_batch(prompts, images_batch, annotations_batch)
                end_time = time.time()
                elapsed_time = end_time - start_time
                self.logger.debug(f"Processed the final batch of {len(prompts)} in {elapsed_time:.2f} seconds.")
        return mmif

def get_app():
    return MoondreamCaptioner()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()
    app = MoondreamCaptioner()
    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
