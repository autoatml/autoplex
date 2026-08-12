"""Adapted pydantic BaseModel for autoplex."""

import logging
from typing import Any

from monty.json import MontyDecoder, jsanitize
from monty.serialization import loadfn
from pydantic import BaseModel, ConfigDict

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class AutoplexBaseModel(BaseModel):
    """Base class for all models in autoplex."""

    model_config = ConfigDict(
        validate_assignment=True,
        protected_namespaces=(),
        extra="allow",
        arbitrary_types_allowed=True,
    )

    def update_parameters(self, updates: dict[str, Any]):
        """
        Update the default parameters of the model instance, including nested fields.

        Args:
            updates (Dict[str, Any]): A dictionary containing the fields as keys to update.
        """
        for key, value in updates.items():
            if hasattr(self, key):
                field_value = getattr(self, key)
                if isinstance(field_value, self.__class__) and isinstance(value, dict):
                    # Update nested model
                    field_value.update_parameters(
                        value
                    )  # Recursively call update_parameters
                else:
                    # Update field value
                    setattr(self, key, value)

            else:
                logging.warning(
                    f"Field {key} not found in default {self.__class__.__name__} model."
                    f"New field has been added. Please ensure the added field contains correct datatype."
                )
                setattr(self, key, value)

    @classmethod
    def from_file(cls, filename: str):
        """
        Load the parameters from a file.

        Args:
            filename (str): The name of the file to load the parameters from.
        """
        custom_params = loadfn(filename)

        return cls(**custom_params)

    def as_dict(self):
        """Return the model as a MSONable dictionary."""
        return jsanitize(
            self.model_copy(deep=True), strict=True, allow_bson=True, enum_values=True
        )

    @classmethod
    def from_dict(cls, d: dict):
        """Create a model from a MSONable dictionary.

        Args:
            d (dict): A MSONable dictionary representation of the Model.
        """
        decoded = {
            k: MontyDecoder().process_decoded(v)
            for k, v in d.items()
            if not k.startswith("@")
        }
        return cls(**decoded)
