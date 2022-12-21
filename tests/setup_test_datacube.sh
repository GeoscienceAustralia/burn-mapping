#!/usr/bin/env bash
set -ex
export METADATA_CATALOG=https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/a4f39b485b33608a016032d9987251881fec4b6f/workspaces/sandbox-metadata.yaml
export PRODUCT_CATALOG=https://raw.githubusercontent.com/GeoscienceAustralia/dea-config/87ca056fa62900596cbf05612da9033fc763009c/workspaces/sandbox-products.csv
export GEOMED_PRODUCT=https://raw.githubusercontent.com/GeoscienceAustralia/burn-mapping/feature/Burn_Cube_AWS_app/dea_burn_cube/configs/gm_products/ga_ls8c_nbart_gm_4cyear_3.odc-product.yaml

# Setup datacube
docker-compose exec -T public_index datacube system init --no-default-types --no-init-users
# Setup metadata types
docker-compose exec -T public_index datacube metadata add "$METADATA_CATALOG"
# Download the product catalog
docker-compose exec -T public_index wget "$PRODUCT_CATALOG" -O product_list.csv
# Add products for testing from the product list
docker-compose exec -T public_index bash -c "tail -n+2 product_list.csv | grep 'ga_ls8c_ard_3' | awk -F , '{print \$2}' | xargs datacube -v product add"

# Setup datacube
docker-compose exec -T private_index datacube system init --no-default-types --no-init-users
# Setup metadata types
docker-compose exec -T private_index datacube metadata add "$METADATA_CATALOG"
# Add the product catalog
docker-compose exec -T private_index datacube product add "$GEOMED_PRODUCT"

# Index test data
cat > ard_index_tiles.sh <<EOF
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/082/2017/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/082/2018/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/082/2019/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/082/2020/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/082/2021/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/083/2017/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/083/2018/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/083/2019/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/083/2020/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/090/083/2021/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/091/082/2017/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/091/082/2018/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/091/082/2019/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/091/082/2020/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
s3-to-dc s3://dea-public-data/baseline/ga_ls8c_ard_3/091/082/2021/*/*/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_ard_3
EOF

cat ard_index_tiles.sh | docker-compose exec -T public_index bash

# Index test data
cat > geomed_index_tiles.sh <<EOF
s3-to-dc s3://dea-public-data-dev/projects/burn_cube/ga_ls8c_nbart_gm_4cyear_3/*/x45/y19/2018--P4Y/*.odc-metadata.yaml --no-sign-request --skip-lineage ga_ls8c_nbart_gm_4cyear_3
EOF

cat geomed_index_tiles.sh | docker-compose exec -T private_index bash
