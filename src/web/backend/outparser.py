from langchain.output_parsers.list import ListOutputParser
import re, json
from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Optional, List, Type, TypeVar
from langchain.schema import BaseOutputParser, OutputParserException
regexp = re.compile("(.*):(.*)",re.MULTILINE|re.DOTALL)
T = TypeVar("T", bound=BaseModel)

PYDANTIC_FORMAT_INSTRUCTIONS = """The output should be formatted as a JSON instance that conforms to the JSON schema below.

Here is the output schema:
```
{schema}
```"""


class MyJsonOutputParser(BaseOutputParser[T]):
        pydantic_object: Type[T]

        def parse(self, text: str) -> T:
            try:
                # Greedy search for 1st json candidate.
                match = re.search(
                    r"\{.*\}", text.strip(), re.MULTILINE | re.IGNORECASE | re.DOTALL
                )
                json_str = ""
                if match:
                    json_str = match.group()
                json_object = json.loads(json_str, strict=False)
                return self.pydantic_object.parse_obj(json_object)

            except (json.JSONDecodeError, ValidationError) as e:
                name = self.pydantic_object.__name__
                msg = f"Failed to parse {name} from completion {text}. Got: {e}"
                raise OutputParserException(msg)

        def get_format_instructions(self) -> str:
            schema = self.pydantic_object.schema()

            # Remove extraneous fields.
            reduced_schema = schema
            if "title" in reduced_schema:
                del reduced_schema["title"]
            if "type" in reduced_schema:
                del reduced_schema["type"]
            # Ensure json in context is well-formed with double quotes.
            schema_str = json.dumps(reduced_schema)

            return PYDANTIC_FORMAT_INSTRUCTIONS.format(schema=schema_str)

        @property
        def _type(self) -> str:
            return "myJsonParser"


class ProteinsListOutputParser(ListOutputParser):
    """Parse out comma separated lists."""

    def get_format_instructions(self) -> str:
        return (
            "Your response should be a list of comma separated values, "
            "eg: `foo, bar, baz`"
        )

    def parse(self, text: str) -> List[Dict]:
        """Parse the output of an LLM call."""
        # print(text)
        lines = text.split("\n")
        lst = []
        for line in lines:
            match = re.match(regexp, line)
            if match:
                virus = match.group(1).strip()
                proteins = match.group(2).strip()
                proteins = re.sub("(, ?and|and)", ",", proteins).strip(".")
                proteins = proteins.strip().split(",")
                if virus and proteins:
                    lst.append({"virus":virus, "proteins":proteins})
        return lst


class Virus(BaseModel):
    name: str = Field(description="name of the virus")
    protein_names: List[str] = Field(description="list names of protein")


class EnvelopProtein(BaseModel):
    description: str = Field(description="The description of the function of the protein")
    label: str = Field(description="the label result should be Yes or No")
    explanation: Optional[str] = Field(description="brief explanation for the label result")


class EnvelopDescProtein(BaseModel):
    label: str = Field(description="the label result should be Yes or No")
    explanation: Optional[str] = Field(description="brief explanation for the label result")


class Definition(BaseModel):
    definition: str = Field(description="The definition of the proteins, it must be abstractive and generalized")


class DefinitionWithExplanation(BaseModel):
    description: str = Field(description="The description of the function of the protein")
    definition: str = Field(description="The definition of the proteins, it must be abstractive and generalized")


class SearchKeywords(BaseModel):
    terms: List[str] = Field(description="the list of bio terms")


class GOTermsKeywords(BaseModel):
    molecular_function: List[str] = Field(description="the list of molecular function of GO terms")
    cellular_component: List[str] = Field(description="the list of cellular component of GO terms")
    biological_process: List[str] = Field(description="the list of biological process of GO terms")

