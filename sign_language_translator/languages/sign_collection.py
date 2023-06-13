from typing import List

class SignCollection:
    def __init__(
        self,
        country: str,
        organization: str,
        collection_id: str,
        labels: List[str],
        persons: List[str],
        cameras: List[str],
    ) -> None:
        # make all these properties
        self.country:str = country
        self.organization: str = organization
        self.collection_id: str = collection_id

        self.labels: List[str] = labels
        self.persons: List[str] = persons
        self.cameras: List[str] = cameras

    def name(self) -> str:
        return "-".join(
            (self.country, self.organization, self.collection_id)
        )
