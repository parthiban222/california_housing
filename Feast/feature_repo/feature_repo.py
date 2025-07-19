
from datetime import timedelta
from feast import BigQuerySource, FeatureView, FeatureService, Entity, ValueType

# Define flower species as entity
housing_entity = Entity(
    name="entity_id",
    description="A ",
    value_type=ValueType.INT64
)

# Define feature view for flower measurements
housing_features = FeatureView(
    name="housing_features",
    entities=[housing_entity],
    ttl=timedelta(weeks=52),  # Time-to-live for features
    source=BigQuerySource(
        table=f"PRACTICE.housing",
        timestamp_field="event_timestamp"
    ),
    tags={"assignment":"week_3"}
)

# Create feature service for one model version
# FeatureService groups features for specific use cases
model_v1 = FeatureService(
    name="feast_model_v1",
    features=[housing_features]
)
