% Copyright 2016-2020 Biomedical Imaging Group Rotterdam, Departments of
% Medical Informatics and Radiology, Erasmus MC, Rotterdam, The Netherlands
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.


function combatmatlab(input)
% Apply ComBat harmonization for Radiomics features based on
% https://github.com/Jfortin1/ComBatHarmonization/tree/master/Matlab

% Determine which settings to use. As long as the settings are in the
% settings folder, only need to provide the name, not the path of the file

% Add the path were the combat scipts are located to the Matlab Path
load(input);

try
    disp('Running ComBat code')
    addpath(genpath(ComBatFolder))

    % Run Combat
    if per_feature
        % Per feature
        data_size = size(datvar);
        data_harmonized = zeros(data_size);
        for i_feat = 1:data_size(1);
            new_feat = combat(datvar(i_feat, :), batchvar, modvar, parvar);
            data_harmonized(i_feat, :) = new_feat;
        end
    else
        % Over all features together
        data_harmonized = combat(datvar, batchvar, modvar, parvar);

    end

    % Write Output .mat file
    save(output, 'data_harmonized')
catch exception
    % Error, write error message as output
    disp(exception)
    disp(output)
    message = exception.message;
    save(output, 'message')
end

exit;

end
