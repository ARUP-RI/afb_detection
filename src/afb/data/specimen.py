from afb.data.lab_id import LabID
from afb.data.m48_scale import M48Scale
from afb.data.metadata import MetaData


# pydantic BaseModel?
class Specimen:
    """
    Represents a physical specimen.
    """

    lab_id: LabID
    sample_type: str
    notes: str

    positive: bool
    mgit_culture_positive: str
    clsi_m48: M48Scale
    kin_pos: bool
    ao_pos: bool
    m48_pos: bool
    ground_truth: str

    def __init__(self, lab_id: LabID):
        self.lab_id = lab_id
        specimen = MetaData.specimen_lookup(lab_id=str(lab_id))

        self.sample_type = specimen.sample_type
        self.notes = specimen.notes
        self.positive = specimen.afb_positive
        self.mgit_culture_positive = specimen.mgit_culture_positive
        self.clsi_m48 = M48Scale.from_str(str(specimen.clsi_m48))
        self.kin_pos = specimen.kin_pos
        self.ao_pos = specimen.ao_pos
        self.m48_pos = specimen.m48_pos
        self.ground_truth = specimen.ground_truth

    def __repr__(self):
        return (
            f"<Specimen: {self.lab_id}"
            f" mgit_culture_positive={self.mgit_culture_positive}"
            f" clsi_m48={self.clsi_m48}"
            f' sample_type="{self.sample_type}"'
            f' notes="{self.notes}'
            f" kin_pos={self.kin_pos}"
            f" ao_pos={self.ao_pos}"
            f" m48_pos={self.m48_pos}"
            f" ground_truth={self.ground_truth}"
            f" positive={self.positive}"
            ">"
        )
