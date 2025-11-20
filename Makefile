build-docker-dist:
	docker build -t bingp/versa-cube:go-dist -f Dockerfile.dist .

launch-docker-dist-sto:
	docker run --privileged --gpus=all -it -v ~/versa-cube:/workspace/versa-cube \
		-v /share/usrdata/zcq/data/:/workspace/data/ \
		-v /share/hdddata/zcq/data/:/workspace/data0/ \
		-v /dev/shm:/dev/shm --network host \
		--hostname vc-dist-$(shell hostname) --name vc-dist-$(shell hostname) \
		bingp/versa-cube:go-dist /bin/bash

launch-docker-dist:
	docker run --privileged --gpus=all -it -v ~/versa-cube:/workspace/versa-cube \
		-v /share/usrdata/zcq/data/:/workspace/data/ \
		-v /dev/shm:/dev/shm --network host \
		--hostname vc-dist-$(shell hostname) --name vc-dist-$(shell hostname) \
		bingp/versa-cube:go-dist /bin/bash

launch-docker-dist-no-sto:
	docker run --privileged --gpus=all -it -v ~/versa-cube:/workspace/versa-cube \
		-v /dev/shm:/dev/shm --network host \
		--hostname vc-dist-$(shell hostname) --name vc-dist-$(shell hostname) \
		bingp/versa-cube:go-dist /bin/bash

launch-docker-spark:
	docker run --privileged --gpus=all -it \
		-v ~/versa-cube/spark-test:/opt/spark/work-dir/test \
		-v /share/usrdata/zcq/data/:/opt/spark/work-dir/data/ \
		-v /share/hdddata/zcq/data/:/opt/spark/work-dir/data0/ \
		--network host --pid=host --ipc=host \
		--name spark-$(shell hostname) bingp/spark-ucx /bin/bash

launch-dhap-nomnt:
	docker run --privileged --gpus=all -it --network host \
		--hostname dhap --name dhap-local \
		bingp/dhap:local-test /bin/bash