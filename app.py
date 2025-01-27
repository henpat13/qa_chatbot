import ast
import json
import os
import re
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_ID = os.getenv("OPENAI_DEPLOYMENT_ID")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")


class FinancialQAAgent:
    def __init__(self, llm):
        """
        Initialise the Financial QA Agent

        Args:
            llm: the LLM used
        """
        self.llm = llm
        self.question_prompt = PromptTemplate(
            input_variables=["question", "context", "table"],
            template=QUESTION_PROMPT_TEMPLATE,
        )
        self.validate_prompt = PromptTemplate(
            input_variables=["question", "reasoning", "actual_answer"],
            template=VALIDATION_PROMPT,
        )

    def extract_table_data(self, sample: Dict[str, Any]) -> str:
        """
        Convert table data to string

        Args:
            sample Dict[str, Any]: A singular entry from the dataset

        Returns:
            str: Table as a string
        """
        try:
            if isinstance(sample["table"], str):
                table_data = ast.literal_eval(sample["table"])
            else:
                table_data = sample["table"]

            df = pd.DataFrame(table_data[1:], columns=table_data[0])
            return df.to_string(index=False)
        except Exception as e:
            print(f"Error extracting table data: {e}")
            return "Unable to parse table data"

    def llm_solve_question(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Solve the question using the LLM with reasoning

        Args:
            sample (Dict[str, Any]): A singular entry from the dataset

        Returns:
            Dict[str, Any]: A dictionary with the answer and reasoning
        """
        question = sample.get("qa").get("question", "")
        answer = sample.get("qa").get("answer", "")
        pre_text = sample.get("pre_text", "")
        post_text = sample.get("post_text", "")
        context = (
            f"Text before the table: {pre_text}\nText after the table: {post_text}"
        )
        table = self.extract_table_data(sample)

        try:
            llm_response = self.llm(
                [
                    SystemMessage(
                        content="Your role is to answer financial mathematical questions."
                    ),
                    HumanMessage(
                        content=self.question_prompt.format(
                            question=question, context=context, table=table
                        )
                    ),
                ]
            )

            try:
                clean_response = re.sub(r"\s+", " ", llm_response.content).strip()
                llm_response_dict = json.loads(clean_response)
            except json.JSONDecodeError:
                try:
                    llm_response_dict = ast.literal_eval(clean_response)
                except (SyntaxError, ValueError):
                    print(f"Could not parse response: {clean_response}")
                    return {
                        "question": question,
                        "error": "Unable to parse LLM response",
                        "actual_answer": answer,
                    }

            # Validate the dictionary has all required keys
            required_keys = [
                "Reasoning_Steps",
                "Relevant_Data_Points",
                "Calculation_Formula",
                "Potential_Validation_Checks",
                "Final_Answer",
                "Confidence_Level",
            ]

            for key in required_keys:
                if key not in llm_response_dict:
                    print(f"Missing key: {key}")
                    return {
                        "question": question,
                        "error": f"Missing required key: {key}",
                        "actual_answer": answer,
                    }

            return {
                "question": question,
                "response": llm_response_dict,
                "actual_answer": answer,
            }

        except Exception as e:
            print("Error in llm_solve_question: ", e)
            return {"question": question, "error": str(e), "actual_answer": answer}

    def validate_answer(self, answer_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Self validation of answer and reasoning

        Args:
            answer_dict (Dict[str, Any]): Dictionary with answer and reasoning

        Returns:
            Dict[str, Any]: Dictionary of validation
        """
        question = answer_dict.get("question", "")
        reasoning = answer_dict.get("response", {})
        actual_answer = answer_dict.get("actual_answer", "")

        try:
            llm_response = self.llm(
                [
                    SystemMessage(
                        content="Your role is to validate the reasoning behind the choice of a calculation."
                    ),
                    HumanMessage(
                        content=self.validate_prompt.format(
                            question=question,
                            reasoning=json.dumps(reasoning),
                            actual_answer=actual_answer,
                        )
                    ),
                ]
            )

            llm_response = llm_response.content
            llm_response_dict = json.loads(llm_response.replace("\n", ""))

            answer_dict["validation_response"] = llm_response_dict
            return answer_dict
        except Exception as e:
            answer_dict["validation_response"] = {"error": str(e)}
            return answer_dict

    def calculate_accuracy_metrics(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, float]]:
        """
        Calculate accuracy metrics for each individual result

        Args:
            results (List[Dict[str, Any]]): List of results from Q&A process

        Returns:
            List[Dict[str, float]]: List of dictionaries with accuracy metrics for each result
        """
        entry_metrics = []

        for result in results:
            # Exact Match Accuracy
            exact_match = int(
                self._parse_answer(
                    str(result.get("response", {}).get("Final_Answer", ""))
                )
                == self._parse_answer(result.get("actual_answer", ""))
            )

            # Fuzzy Match Accuracy
            fuzzy_match = int(
                self._fuzzy_match(
                    self._parse_answer(
                        str(result.get("response", {}).get("Final_Answer", ""))
                    ),
                    self._parse_answer(result.get("actual_answer", "")),
                )
            )

            # Validation Confidence
            validation_confidence = int(
                result.get("validation_response", {}).get("Validity_Assessment", "")
                == "High"
            )

            entry_metrics.append(
                {
                    "exact_match": exact_match,
                    "fuzzy_match": fuzzy_match,
                    "validation_confidence": validation_confidence,
                }
            )

        return entry_metrics

    def _parse_answer(self, answer: str) -> str:
        """
        Parse and clean numerical answers

        Args:
            answer str: Raw answer string

        Returns:
            str: Cleaned numerical answer
        """
        # Extract numerical values, handle percentage
        matches = re.findall(r"-?\d+\.?\d*", answer)
        return matches[0] if matches else ""

    def _fuzzy_match(self, pred: str, actual: str, tolerance: float = 0.1) -> bool:
        """
        Perform fuzzy numerical matching

        Args:
            pred: Predicted answer
            actual: Actual answer
            tolerance: Acceptable percentage difference

        Returns:
            Boolean indicating if answers are close enough
        """
        try:
            pred_float = float(pred)
            actual_float = float(actual)
            diff = abs(pred_float - actual_float) / abs(actual_float)
            return diff <= tolerance
        except ValueError:
            return False


def process_financial_qa_dataset(
    dataset_path, model, output_csv_path="financial_qa_results.csv"
):
    """
    Process the entire financial QA dataset and save results

    Args:
        dataset_path (str): Path to the input dataset
        model: Language model to use
        output_csv_path (str): Path to save the output CSV

    Returns:
        pandas.DataFrame: Processed dataset
    """
    # Load dataset
    with open(dataset_path, "r") as file:
        dataset = json.load(file)

    # Only using the first 10 entries for time purposes
    dataset = dataset[:10]

    agent = FinancialQAAgent(llm=model)

    results = []
    processed_results = []

    for sample in dataset:
        try:
            qa_entries = {}

            if "qa" in sample:
                qa_entries["qa"] = sample["qa"]

            qa_keys = [key for key in sample.keys() if key.startswith("qa_")]
            for key in qa_keys:
                qa_entries[key] = sample[key]

            for qa_key, qa_entry in qa_entries.items():
                modified_sample = sample.copy()
                modified_sample["qa"] = qa_entry
                solved_question = agent.llm_solve_question(modified_sample)
                validated_question = agent.validate_answer(solved_question)

                result_entry = {
                    "id": sample.get("id", ""),
                    "qa_key": qa_key,
                    "question": validated_question.get("question", ""),
                    "actual_answer": validated_question.get("actual_answer", ""),
                    # Reasoning steps
                    "reasoning_steps": validated_question.get("response", {}).get(
                        "Reasoning_Steps", ""
                    ),
                    "relevant_data_points": validated_question.get("response", {}).get(
                        "Relevant_Data_Points", ""
                    ),
                    "calculation_formula": validated_question.get("response", {}).get(
                        "Calculation_Formula", ""
                    ),
                    "potential_validation_checks": validated_question.get(
                        "response", {}
                    ).get("Potential_Validation_Checks", ""),
                    "final_answer": validated_question.get("response", {}).get(
                        "Final_Answer", ""
                    ),
                    "confidence_level": validated_question.get("response", {}).get(
                        "Confidence_Level", ""
                    ),
                    # Validation information
                    "validity_assessment": validated_question.get(
                        "validation_response", {}
                    ).get("Validity_Assessment", ""),
                    "potential_issues": validated_question.get(
                        "validation_response", {}
                    ).get("Potential_Issues", ""),
                    "suggested_improvements": validated_question.get(
                        "validation_response", {}
                    ).get("Suggested_Improvements", ""),
                    "validation_confidence": validated_question.get(
                        "validation_response", {}
                    ).get("Confidence_Level", ""),
                }

                processed_results.append(result_entry)
                results.append(validated_question)

        except Exception as e:
            print(f"Error processing sample: {e}")

    # Calculate accuracy metrics
    accuracy_metrics = agent.calculate_accuracy_metrics(results)

    df = pd.DataFrame(processed_results)

    df["exact_match_accuracy"] = [metric["exact_match"] for metric in accuracy_metrics]
    df["fuzzy_match_accuracy"] = [metric["fuzzy_match"] for metric in accuracy_metrics]
    df["validation_confidence"] = [
        metric["validation_confidence"] for metric in accuracy_metrics
    ]

    # Save to CSV
    df.to_csv(output_csv_path, index=False)

    return df


def convert_to_serialisable(obj):
    """
    Convert numpy types to standard Python types for JSON serialisation
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def perform_statistical_analysis(df):
    """
    Perform statistical analysis on the results dataset

    Args:
        df (pandas.DataFrame): Dataset to analyse

    Returns:
        dict: Dictionary of statistical insights
    """
    accuracy_columns = [
        "exact_match_accuracy",
        "fuzzy_match_accuracy",
        "validation_confidence",
    ]

    stats_summary = {}
    for col in accuracy_columns:
        try:

            def _clean_value(x):
                str_x = str(x)
                matches = re.findall(r"-?\d+\.?\d*", str_x)
                return float(matches[0]) if matches else 0

            df[col] = df[col].apply(_clean_value)

            stats_summary[col] = {
                "mean": convert_to_serialisable(df[col].mean()),
                "total_count": len(df),
                "success_count": convert_to_serialisable(df[col].sum()),
                "success_percentage": convert_to_serialisable(df[col].mean() * 100),
                "valid_entries": len(df),
            }
        except Exception as e:
            print(f"Error processing column {col}: {e}")
            stats_summary[col] = {"error": str(e)}

    return stats_summary


def main(dataset_path, model):
    """
    Main function to run the entire analysis pipeline

    Args:
        dataset_path (str): Path to the input dataset
        model: Language model to use
    """
    df = process_financial_qa_dataset(dataset_path, model)
    stats_summary = perform_statistical_analysis(df)

    with open("performance_analysis_results.json", "w") as f:
        json.dump(stats_summary, f, indent=2)


QUESTION_PROMPT_TEMPLATE = """
Context Information:
{context}

Table Information:
{table}

Question:
{question}

Your task:
1. Carefully analyse the question and the context
2. Read through the table information in depth
3. Solve the question with clear, step-by-step reasoning
4. IMPORTANT: Provide your solution as a VALID JSON dictionary. Ensure:
   - No trailing commas
   - All strings are in double quotes
   - No comments
   - Proper JSON syntax

{{"Reasoning_Steps": A detailed explanation of how you approached the problem,
"Relevant_Data_Points": Key numbers and data you used to calculate the answer,
"Calculation_Formula": The mathematical formula or logic used to calculate the answer,
"Potential_Validation_Checks": Ways to verify the answer,
"Final_Answer": State your answer which should be a number
"Confidence_Level": Estimate your confidence in the answer from 0-100%
}}

Only return the valid JSON dictionary.
"""

VALIDATION_PROMPT = """
Question:
{question}

Reasoning:
{reasoning}

Actual Answer:
{actual_answer}

Your task:
1. Assess the question and the reasoning for the calculated answer
2. Compare the reasoning with the actual answer
3. Identify any potential errors
4. Suggest improvements or different approaches
5. IMPORTANT: Provide an overall assessment as a VALID JSON dictionary. Ensure:
   - No trailing commas
   - All strings are in double quotes
   - No comments
   - Proper JSON syntax

{{"Validity_Assessment": High/Medium/Low,
"Potential_Issues": List any problems,
"Suggested_Improvements": Recommendations for improvements,
"Confidence_Level": Estimate your confidence in your validation from 0-100%}}

Only return the valid JSON dictionary.
"""

if __name__ == "__main__":

    model = AzureChatOpenAI(
        azure_endpoint=OPENAI_API_BASE,
        azure_deployment=OPENAI_DEPLOYMENT_ID,
        api_version=OPENAI_API_VERSION,
        model_name=OPENAI_DEPLOYMENT_ID,
        api_key=OPENAI_API_KEY,
        temperature=0,
    )

    dataset_path = "train.json"
    main(dataset_path, model)
