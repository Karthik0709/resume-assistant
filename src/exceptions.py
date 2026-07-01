class RagException(Exception) :
    """Base Exception for any specific exceptions in this project"""
    pass

class DocumentLoadingError(RagException):
    """ Raised when a document fails to load or parse """
    pass

class EmbeddingError(RagException):
    """ Raised when embedding model fails to load or parse """
    pass

class VectorStoreError(RagException):
    """Raised when the vector database operation fails"""
    pass

class RetrievalError(RagException):
    """Raised when retrieval operation from vectorstore fails"""
    pass

class LLMError(RagException):
    """Raised when LLM connection or response related errors occur"""
    pass