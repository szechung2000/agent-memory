"""add source-scoped external ids for memory upserts

Revision ID: a9a280afb5d5
Revises: f61a6110a7e6
Create Date: 2026-08-28
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "a9a280afb5d5"
down_revision: str | Sequence[str] | None = "f61a6110a7e6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "memories" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("memories")}
    if "external_id" not in columns:
        op.add_column("memories", sa.Column("external_id", sa.String(length=255), nullable=True))
    indexes = {index["name"] for index in inspector.get_indexes("memories")}
    if "uq_memories_user_namespace_external_id" not in indexes:
        op.create_index(
            "uq_memories_user_namespace_external_id",
            "memories",
            ["user_id", "namespace", "external_id"],
            unique=True,
        )


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if "memories" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("memories")}
    indexes = {index["name"] for index in inspector.get_indexes("memories")}
    if "uq_memories_user_namespace_external_id" in indexes:
        op.drop_index("uq_memories_user_namespace_external_id", table_name="memories")
    if "external_id" in columns:
        op.drop_column("memories", "external_id")
