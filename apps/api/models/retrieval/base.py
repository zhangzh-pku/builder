from .models.PDF import PDFRetrieverMixin
from .models.retriever import Retriever as ImportedRetriever

class Retriever(ImportedRetriever):
    pass
