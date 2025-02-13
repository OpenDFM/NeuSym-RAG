#coding=utf8
from typing import Any, List, Dict, Tuple, Optional
import base64
from utils.functions.image_functions import get_image_message

def formulate_input(
        dataset: str, 
        data: Dict[str, Any],
        use_reference_pdf: bool = True,
        image_limit: int = 10
) -> Tuple[str, str, str, Optional[Dict[str, Any]]]:
    if dataset == 'pdfvqa':
        question, page = data['question'], data['page_number']
        pdf_id = data['pdf_id']
        question += f" (for page {page} in PDF with id {pdf_id})" if page is not None else f" (for PDF with id {pdf_id})"
        question_type = data['question_type']
        if question_type == 'existence':
            answer_format = 'Your answer should be either "True" or "False" without punctuation.'
        elif question_type == 'counting':
            answer_format = 'Your answer should be a single integer number, e.g., 1, 2, 3, etc.'
        elif question_type in ['object_recognition', 'structural_understanding']:
            answer_format = 'Your answer should be verbose text from the raw PDF, e.g., figure/table captions, section titles, or "No specific Section." if not found. Note that, for questions relevant to figures or tables, the answers are mostly the captions or paragraphs surrounding them.'
        elif question_type in ['parent_relationship_understanding', 'child_relationship_understanding']:
            answer_format = 'Your answer should be a Python list of strings in the following format: ["str1", "str2", ...], where each string represents mostly a section or subsection title, do not ignore the double quotes. Note that, some parent-children belongingships are not recorded and you may need to infer from the position or text content of different regions or bounding boxes. If not found, please return the list ["No Section!"] or ["No subsection!"].'
        else:
            raise NotImplementedError(f"Question type {question_type} not supported.")
    elif dataset == 'tatdqa':
        question, page = data['question'], data['page_number']
        pdf_id = data['pdf_id']
        question += f" (for page {page} in PDF with id {pdf_id})" if page is not None else f" (for PDF with id {pdf_id})"
        question_type = data['question_type']
        scale = data['answer'][1]
        if question_type == 'count':
            answer_format = 'Your answer should be a single integer number, e.g., 1, 2, 3, etc.'
        elif question_type in ['span', 'multi-span', 'arithmetic']:
            if scale != '' and question_type == 'multi-span':
                answer_format = 'Your answer should be in the following Python list format: [[answer1, answer2, ...], scale]. Note that each `answer` can be either str or float, `scale` should be one of the following: "percent", "thousand", "million" and "". Remember that: 1. even if you don\'t need a scale, use "" as `scale`; 2. do not ignore the double quotes and the brackets.'
            elif scale != '' or question_type == 'arithmetic':
                answer_format = 'Your answer should be in the following Python list format: [answer, scale]. Note that `answer` can be either int, str or float, `scale` should be one of the following: "percent", "thousand", "million" and "". Remember that: 1. even if you don\'t need a scale, use "" as `scale`; 2. do not ignore the double quotes and the brackets.'
            elif question_type == 'multi-span':
                answer_format = 'You should directly print your answer in the following Python list format: ["part1", "part2", ...], do not ignore the outer brackets and inner double quotes. Remember that: 1. even the answer is a single text string, please wrap it with a Python list of length one; 2. for an extremely long text answer, try to divide it into multiple concise bullet points and place them into the Python list; 3. FOR EACH ELEMENT IN THE PYTHON LIST, RETRIEVE IT FROM THE RAW PDF, AND MAKE IT AS CONCISE AS POSSIBLE.'
            elif question_type == 'span':
                answer_format = 'Your answer should be verbose text from the raw PDF. RETRIEVE IT FROM THE RAW PDF, AND MAKE IT AS CONCISE AS POSSIBLE.'
        else:
            raise NotImplementedError(f"Question type {question_type} not supported.")
    else:
        question, answer_format, pdf_context, image_message = data['question'], data['answer_format'], "", None
        anchor_pdf_id = data.get('anchor_pdf', 'pdf_id')
        assert isinstance(anchor_pdf_id, list), f"Example {data['uuid']} anchor_pdf should be a list."
        if len(anchor_pdf_id) == 1:
            pdf_context += f"[Anchor PDF]: {anchor_pdf_id[0]}\n"
        elif len(anchor_pdf_id) > 1:
            pdf_context += f"[Anchor PDFs]: [{', '.join(anchor_pdf_id)}]\n"
        
        use_reference_pdf = use_reference_pdf if dataset != 'm3sciqa' else False
        if use_reference_pdf and data.get('reference_pdf', '') :
            reference_pdf_id = data.get('reference_pdf', '')
            assert isinstance(reference_pdf_id, list), f"Example {data['uuid']} reference_pdf should be a list."
            if len(reference_pdf_id) == 1:
                pdf_context += f"[Reference PDF]: {reference_pdf_id[0]}\n"
            elif len(reference_pdf_id) > 1:
                pdf_context += f" [Reference PDFs]: [{', '.join(reference_pdf_id)}]\n"

        if image_limit > 0 and data.get('anchor_image', []):
            template = "[Image]: Here are some images that you can use to answer the question:" if len(data['anchor_image']) > 1 else "[Image]: Here is one image you can use to answer the question:"
            image_message = get_image_message(
                template=template,
                image_path=data['anchor_image'],
                image_limit=image_limit
            )
        return question, answer_format, pdf_context, image_message
    return question, answer_format