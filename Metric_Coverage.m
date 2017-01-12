function Coverage = Metric_Coverage(Outputs, test_target)
%Computing the coverage
%Outputs: the predicted outputs of the classifier, the output of the ith instance for the jth class is stored in Outputs(j,i)
%test_target: the actual labels of the test instances, if the ith instance belong to the jth class, test_target(j,i)=1, otherwise test_target(j,i)=-1

       [num_class, num_instance] = size(Outputs);
    
       Label = cell(num_instance,1);      %ÿ�������������еı�ǩ����
       not_Label = cell(num_instance,1);  %ÿ���������������еı�ǩ����
       Label_size = zeros(1,num_instance);  %ÿ�������������еı�ǩ��
       for i = 1:num_instance
           temp = test_target(:,i);
           Label_size(1,i) = sum(temp==ones(num_class,1));
           for j = 1:num_class
               if(temp(j)==1)
                   Label{i,1} = [Label{i,1},j];
               else
                   not_Label{i,1} = [not_Label{i,1},j];
               end
           end
       end

       cover = 0;
       for i = 1:num_instance
           temp = Outputs(:,i);
           [~,index] = sort(temp);  %���յ�i���������и�����ǩ�ĸ��ʴ�С���� �������б�ǩ��������index��
           temp_min = num_class+1;
           for m = 1:Label_size(i)
               [~,loc] = ismember(Label{i,1}(m),index);  %��i�����������ı�ǩ���е�m����ǩ��Ӧindex�е�λ��
               if(loc<temp_min)
                   temp_min = loc;  %��i�����������ı�ǩ������Ӧindex�е���Сλ�ã�Ҳ�����ҳ�������ȫ�����б�ǩ����Ҫ�����
               end
           end
           cover = cover+(num_class-temp_min+1);   %�ۼ�ÿ���������������
       end
       Coverage = (cover/num_instance)-1;    %ƽ�����
       