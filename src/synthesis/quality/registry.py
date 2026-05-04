"""Database and schema registry abstractions for quality control."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .models import DatabaseSchema


class DatabaseClient(Protocol):
    def inspect_schema(self) -> DatabaseSchema:
        ...

    def execute_read_only(self, sql: str, *, max_preview_rows: int) -> tuple[int, list[dict[str, object]]]:
        ...


class DatabaseRegistry(Protocol):
    def get_client(self, database_id: str) -> DatabaseClient:
        ...


class SchemaRegistry(Protocol):
    def get_schema(self, database_id: str) -> DatabaseSchema | None:
        ...

    def set_schema(self, schema: DatabaseSchema) -> None:
        ...


@dataclass
class InMemorySchemaRegistry:
    schemas: dict[str, DatabaseSchema] = field(default_factory=dict)

    def get_schema(self, database_id: str) -> DatabaseSchema | None:
        return self.schemas.get(database_id)

    def set_schema(self, schema: DatabaseSchema) -> None:
        self.schemas[schema.database_id] = schema


@dataclass
class StaticDatabaseRegistry:
    clients: dict[str, DatabaseClient]

    def get_client(self, database_id: str) -> DatabaseClient:
        if database_id not in self.clients:
            raise KeyError(f"Database client not registered: {database_id}")
        return self.clients[database_id]

